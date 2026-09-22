//! CPU topology detection and thread affinity for asymmetric multi-core systems (e.g. big.LITTLE SBCs).

use anyhow::Result;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Returns the logical CPU IDs of the performance (big) cores.
///
/// On Linux:
/// Inspects `/sys/devices/system/cpu/cpu*/cpu_capacity` and
/// `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq` to identify the
/// performance cores, intersecting with the current process affinity mask.
///
/// On macOS:
/// Queries `hw.perflevel0.logicalcpu`.
pub fn performance_cores() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        let allowed = get_allowed_cores();
        if allowed.is_empty() {
            return (0..std::thread::available_parallelism().map_or(1, usize::from)).collect();
        }

        // 1. Try reading cpu_capacity from sysfs
        let mut capacities = Vec::new();
        for &cpu in &allowed {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
            if let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(cap) = content.trim().parse::<u64>()
            {
                capacities.push((cpu, cap));
            }
        }
        if !capacities.is_empty() {
            let max_cap = capacities.iter().map(|(_, c)| *c).max().unwrap_or(0);
            let big_cores: Vec<usize> = capacities
                .into_iter()
                .filter(|(_, c)| *c == max_cap)
                .map(|(cpu, _)| cpu)
                .collect();
            if !big_cores.is_empty() {
                return big_cores;
            }
        }

        // 2. Try reading cpuinfo_max_freq from sysfs
        let mut freqs = Vec::new();
        for &cpu in &allowed {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
            if let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(freq) = content.trim().parse::<u64>()
            {
                freqs.push((cpu, freq));
            }
        }
        if !freqs.is_empty() {
            let max_freq = freqs.iter().map(|(_, f)| *f).max().unwrap_or(0);
            let big_cores: Vec<usize> = freqs
                .into_iter()
                .filter(|(_, f)| *f == max_freq)
                .map(|(cpu, _)| cpu)
                .collect();
            if !big_cores.is_empty() {
                return big_cores;
            }
        }

        allowed
    }

    #[cfg(target_os = "macos")]
    {
        let count = macos_performance_core_count()
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
        (0..count).collect()
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let count = std::thread::available_parallelism().map_or(1, usize::from);
        (0..count).collect()
    }
}

/// Default number of worker threads for the CPU pool.
///
/// On asymmetric systems, returns the number of performance cores (e.g. 4 on RK3588).
pub fn default_threads() -> usize {
    let cores = performance_cores();
    if !cores.is_empty() {
        cores.len()
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    }
}

#[cfg(target_os = "linux")]
fn get_allowed_cores() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            let mut cores = Vec::new();
            for i in 0..libc::CPU_SETSIZE as usize {
                if libc::CPU_ISSET(i, &set) {
                    cores.push(i);
                }
            }
            if !cores.is_empty() {
                return cores;
            }
        }
    }
    (0..std::thread::available_parallelism().map_or(1, usize::from)).collect()
}

/// Sets thread affinity of the current thread to a specific CPU core.
#[cfg(target_os = "linux")]
pub fn set_core_affinity(cpu_id: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu_id, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn set_core_affinity(_cpu_id: usize) {}

/// Pins the current thread to the performance cores.
#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_performance_cores() {
    let cores = performance_cores();
    if !cores.is_empty() {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in &cores {
                libc::CPU_SET(cpu, &mut set);
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread_to_performance_cores() {}

#[cfg(target_os = "macos")]
fn macos_performance_core_count() -> Option<usize> {
    let mut count: libc::c_int = 0;
    let mut size = std::mem::size_of_val(&count);
    // SAFETY: sysctlbyname writes at most `size` bytes into `count`.
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            std::ptr::from_mut(&mut count).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && count > 0).then_some(count as usize)
}

/// Set high priority / QoS on the current thread.
pub fn set_high_priority() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // QOS_CLASS_USER_INTERACTIVE = 0x21
        pthread_set_qos_class_self_np(0x21, 0);
    }
}

/// Builds a Rayon thread pool optimized for CPU inference.
///
/// Worker threads are pinned to performance cores on Linux and given high priority.
pub fn create_cpu_pool(threads: usize) -> Result<ThreadPool> {
    let perf_cores = performance_cores();
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|n| format!("hy-cpu-{n}"))
        .start_handler(move |idx| {
            set_high_priority();
            if !perf_cores.is_empty() {
                set_core_affinity(perf_cores[idx % perf_cores.len()]);
            }
        })
        .build()
        .map_err(Into::into)
}
