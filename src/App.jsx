import { useState, useCallback } from "react";
import "./App.css";
import { useModel } from "./hooks/useModel";

function App() {
    const [webgpuEnabled, setWebgpuEnabled] = useState(false);

    const { run } = useModel();

    const onRun = useCallback(async () => {

        await run(webgpuEnabled);
        
    }, [run, webgpuEnabled]);

    return (
        <main style={{ display: "flex", flexDirection: "column" }}>
            <button onClick={onRun}>Run Demucs</button>
            <div>
                <input
                    type="checkbox"
                    onChange={(e) => setWebgpuEnabled(e.target.checked)}
                />
                <label htmlFor="webgpu">Enable WebGPU</label>
            </div>
        </main>
    );
}

export default App;
