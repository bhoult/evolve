fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("adapter");
    let info = adapter.get_info();
    let l = adapter.limits();
    println!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
    println!("max_buffer_size {} MB", l.max_buffer_size >> 20);
    println!("max_storage_buffer_binding_size {} MB", l.max_storage_buffer_binding_size >> 20);
    println!("max_compute_workgroups_per_dimension {}", l.max_compute_workgroups_per_dimension);
    println!("timestamp query: {}", adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY));
    println!("shader f16: {}", adapter.features().contains(wgpu::Features::SHADER_F16));
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}
