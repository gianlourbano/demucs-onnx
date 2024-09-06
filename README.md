# Demucs inference on ONNX runtime

This repository contains a simple example of how to run inference on HTDemucs model using ONNX runtime. The model is first converted to ONNX format using the new dynamo `torch.export` (as described [here](https://github.com/gianlourbano/demucs)) and then loaded into ONNX runtime for inference.

## Usage

Run `npm run dev` to start the vite development server. Run the model, check errors in the console.

