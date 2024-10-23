#pragma once

#include <numeric>

#include "ONNX_Base.h"

class ASCRecog : ONNX_Base {
private:
	cv::Mat Preprocessing(std::string imagePath);
	cv::Mat Preprocessing(cv::Mat);
	char GetArgMax();


public:
	ASCRecog(cv::Size inputSZ = cv::Size(64, 64),
		cv::Size outputSZ = cv::Size(0, 0),
		std::vector<const char*> inNodes = { "posts" },
		std::vector<const char*> outNodes = { "ASC_OUT" })
		: ONNX_Base(inputSZ, outputSZ, inNodes, outNodes)
	{
		auto tmpEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ASCRecog");
		mEnv = std::move(tmpEnv);
		//HAN_RECOG HAN_RECOG_FILE
		auto embedData = GetEmbedData(ASC_RECOG, ASC_RECOG_FILE);
		this->LoadFromBuffer(embedData);
	};

	ASCRecog(ASCRecog&&) = delete;

	void Test(std::string imagePath);
	char GetResult(std::string);
	char GetResult(cv::Mat);

	void WordTypeCorrection();

	
};