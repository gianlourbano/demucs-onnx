import { useState, useCallback } from "react";
import "./App.css";
import { useModel } from "./hooks/useModel";

function App() {
    const [webgpuEnabled, setWebgpuEnabled] = useState(false);
    const [optimized, setOptimized] = useState(false);

    const { run } = useModel();

    const onRun = useCallback(async () => {

        await run(webgpuEnabled, optimized);
        
    }, [run, webgpuEnabled, optimized]);

    return (
        <main style={{ display: "flex", flexDirection: "column" }}>
            <button onClick={onRun}>Run Demucs</button>
            <div>
                <input
                    id="webgpu"
                    type="checkbox"
                    onChange={(e) => setWebgpuEnabled(e.target.checked)}
                />
                <label htmlFor="webgpu">Enable WebGPU</label>
                <input
                    id="optimized"
                    type="checkbox"
                    onChange={(e) => setOptimized(e.target.checked)}
                />
                <label htmlFor="optimized">Use optimized model</label>
            </div>
        </main>
    );
}

export default App;
