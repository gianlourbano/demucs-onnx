const load = async (webgpu) => {
    const isWebGPUavailable = "gpu" in navigator;

    if (webgpu && !isWebGPUavailable) {
        console.warn("WebGPU is not available in this browser.");
    }

    const runtime =
        webgpu && isWebGPUavailable
            ? await import("onnxruntime-web/webgpu")
            : await import("onnxruntime-web");

    runtime.env.wasm.numThreads = Math.max(
        1,
        navigator.hardwareConcurrency - 2
    );

    return runtime;
};

const no_chunks = 1;
const audioChunks = [];
const specs = [];

for (let i = 0; i < no_chunks; i++) {
    audioChunks.push(new Float32Array(1 * 2 * 441000));
    specs.push(new Float32Array(1 * 2 * 2048 * 431 * 2));
}

for (let i = 0; i < no_chunks; i++) {
    // fill with random values
    for (let j = 0; j < audioChunks[i].length; j++) {
        audioChunks[i][j] = Math.random();
    }
    for (let j = 0; j < specs[i].length; j++) {
        specs[i][j] = Math.random();
    }
}

export const useModel = () => {
    return {
        run: async (webgpu, profiling) => {
            const { InferenceSession, Tensor, env } = await load(webgpu);

            const model = "demucs.onnx";
            console.log("Using model: ", model);

            const model_binary = await fetch(model);
            const model_uint8 = new Uint8Array(
                await model_binary.arrayBuffer()
            );

            console.time("onnx");
            const startTime = performance.now();

            const session = await InferenceSession.create(model_uint8, {
                executionProviders: webgpu ? ["webgpu"] : undefined,
                //enableProfiling: false,
            });

            let wgpu_profile = "[\n";
            if (profiling) {
                env.webgpu.profiling = {
                    mode: "default",
                    ondata: (data) => {
                        wgpu_profile += JSON.stringify(data) + ",\n";
                    },
                };
            }

            let inputs = [];
            for (let i = 0; i < audioChunks.length; i++) {
                inputs.push({
                    [session.inputNames[0]]: new Tensor(
                        "float32",
                        audioChunks[i],
                        [1, 2, 441000]
                    ),
                    [session.inputNames[1]]: new Tensor(
                        "float32",
                        specs[i],
                        [1, 2, 2048, 431, 2]
                    ),
                });
            }

            for (let idx = 0; idx < inputs.length; idx++) {
                console.time(`step onnx ${idx}`);
                const output = await session.run(inputs[idx], {});
                console.log(output);
                console.timeEnd(`step onnx ${idx}`);
            }

            //session.endProfiling();

            await session.release();

            console.timeEnd("onnx");
            const endTime = performance.now();

            if (profiling) {
                const totalOnnxTime = endTime - startTime;

                wgpu_profile = wgpu_profile.slice(0, -2) + "\n]";
                const profileWgpu = JSON.parse(wgpu_profile);

                const output = profileWgpu.map((x) => {
                    const time = x.endTime - x.startTime;
                    const name = x.kernelName;
                    const type = x.kernelType;

                    //time is in microseconds, convert to milliseconds
                    return { time: time / 1000, name, type };
                });

                const kernelMap = new Map();

                output.forEach((x) => {
                    if (!kernelMap.has(x.type)) {
                        kernelMap.set(x.type, []);
                    }
                    kernelMap.get(x.type).push(x);
                });

                const kernelStats = Array.from(kernelMap.entries()).map(
                    ([name, data]) => {
                        const total = data.reduce((acc, x) => acc + x.time, 0);
                        const count = data.length;
                        const avg = total / count;
                        const percentage = (total / totalOnnxTime / 1000) * 100;

                        return {
                            name,
                            total: parseFloat(total.toFixed(2)),
                            count,
                            avg: parseFloat(avg.toFixed(2)),
                            percentage: `${parseFloat(percentage.toFixed(2))}%`,
                        };
                    }
                );

                const sortedKernelStats = kernelStats.sort(
                    (a, b) => b.total - a.total
                );

                const nodesMap = new Map();

                output.forEach((x) => {
                    if (!nodesMap.has(x.name)) {
                        nodesMap.set(x.name, []);
                    }
                    nodesMap.get(x.name).push(x);
                });

                const nodesStats = Array.from(nodesMap.entries()).map(
                    ([name, data]) => {
                        const total = data.reduce((acc, x) => acc + x.time, 0);
                        const count = data.length;
                        const avg = total / count;
                        const percentage = (total / totalOnnxTime / 1000) * 100;

                        return {
                            name,
                            total: parseFloat(total.toFixed(2)),
                            count,
                            avg: parseFloat(avg.toFixed(2)),
                            percentage: `${parseFloat(percentage.toFixed(2))}%`,
                        };
                    }
                );

                const nodesStatsSorted = nodesStats.sort(
                    (a, b) => b.total - a.total
                );

                console.log(
                    JSON.stringify(sortedKernelStats.slice(0, 10), null, 2)
                );

                console.log(
                    JSON.stringify(nodesStatsSorted.slice(0, 10), null, 2)
                );
            }
        },
    };
};
