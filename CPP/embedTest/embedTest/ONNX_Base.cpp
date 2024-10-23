#include "ONNX_Base.h"

void ONNX_Base::GetModelInfo() {
	const size_t num_input_nodes = (*mSession).GetInputCount();
	std::vector<Ort::AllocatedStringPtr> input_names_ptr;
	input_names_ptr.reserve(num_input_nodes);
	mInputNodeNames.reserve(num_input_nodes);

	for (size_t i = 0; i < num_input_nodes; i++) {
		auto input_name = (*mSession).GetInputNameAllocated(i, mAllocator);
		mInputNodeNames.push_back(input_name.get());
		input_names_ptr.push_back(std::move(input_name));

		auto type_info = (*mSession).GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		mInputNodeDims = tensor_info.GetShape();
	}
}

int ONNX_Base::LoadFromBuffer(std::pair<std::vector<UCHAR>, DWORD> buffInfo) {
	//auto embedData = GetEmbedData(DETECTION, DETECTION_FILE);
	auto tmpSession = std::make_unique<Ort::Session>(*mEnv, &(buffInfo.first[0]), buffInfo.second, mSessionOptions);
	
	mSession = std::move(tmpSession);

	GetModelInfo();
}

int ONNX_Base::LoadFromFile(std::string modelPath) {
	mModelPath = modelPath;

	std::wstring wideStr;
	wideStr.assign(modelPath.begin(), modelPath.end());

	auto tmpSession = std::make_unique<Ort::Session>(*mEnv, wideStr.c_str(), mSessionOptions);
	mSession = std::move(tmpSession);

	GetModelInfo();
}


void ONNX_Base::Inference(std::function<cv::Mat(std::string)> funcPtr, std::string imagePath) {
	cv::Mat blob = funcPtr(imagePath);

	size_t inputTensorSize = kInputImageSize.width * kInputImageSize.height * kChannels;

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, inputTensorSize,
		mInputNodeDims.data(), mInputNodeDims.size());

	assert(input_tensor.IsTensor());

	mOutputTensors =
		(*mSession).Run(Ort::RunOptions{ nullptr }, kTmpInputNodeNames.data(), &input_tensor, 1, kDetectOutputNodeNames.data(), kDetectOutputNodeNames.size());

	assert(mOutputTensors.size() == 1 && mOutputTensors.front().IsTensor());
}

void ONNX_Base::Inference(cv::Mat blob) {
	size_t inputTensorSize = kInputImageSize.width * kInputImageSize.height * kChannels;
	//size_t inputTensorSize = 64 * 64 * 3;

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, inputTensorSize,
		mInputNodeDims.data(), mInputNodeDims.size());

	assert(input_tensor.IsTensor());

	mOutputTensors =
		(*mSession).Run(Ort::RunOptions{ nullptr }, kTmpInputNodeNames.data(), &input_tensor, 1, kDetectOutputNodeNames.data(), kDetectOutputNodeNames.size());

	assert(mOutputTensors.size() == 1 && mOutputTensors.front().IsTensor());
}