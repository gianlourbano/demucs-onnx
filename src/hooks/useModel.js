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

export const useModel = () => {
    return {
        run: async (webgpu) => {
            const { InferenceSession, Tensor } = await load(webgpu);

            const model_binary = await fetch("htdemucs.onnx");
            const model_uint8 = new Uint8Array(
                await model_binary.arrayBuffer()
            );

            const session = await InferenceSession.create(model_uint8, {
                executionProviders: webgpu ? ["webgpu"] : undefined,
                enableProfiling: false,
            });

            // simulating 5 audio chunks with their corresponding spectrograms
            let audioChunks = [];
            let specs = [];
            for (let i = 0; i < 5; i++) {
                // push both mix and spec
                audioChunks.push(new Float32Array(1 * 2 * 441000));
                specs.push(new Float32Array(1 * 2 * 2048 * 431 * 2));

                // fill with random values
                for (let j = 0; j < audioChunks[i].length; j++) {
                    audioChunks[i][j] = Math.random();
                }
                for (let j = 0; j < specs[i].length; j++) {
                    specs[i][j] = Math.random();
                }
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
                console.timeEnd(`step onnx ${idx}`);
            }

            await session.release();

            console.timeEnd("onnx");
        },
    };
};
