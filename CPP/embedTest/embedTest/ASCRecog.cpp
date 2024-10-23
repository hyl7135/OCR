#include "ASCRecog.h"

cv::Mat ASCRecog::Preprocessing(std::string imagePath) {
	mInputImage = cv::imread(imagePath);

	cv::Mat blob;
	cv::resize(mInputImage, blob, cv::Size(64, 64));

	blob.convertTo(blob, CV_32FC3);

	blob = blob / (float)255;

	return blob;
}

cv::Mat ASCRecog::Preprocessing(cv::Mat mat) {
	//cv::Mat blob = mat.clone();
	cv::Mat blob = mat;
	cv::resize(blob, blob, cv::Size(64, 64));

	blob.convertTo(blob, CV_32FC3);

	blob = blob / (float)255;

	return blob;
}

void ASCRecog::Test(std::string imagePath) {
	//Inference(Preprocessing(imagePath));

	//GetResult();
}

char ASCRecog::GetResult(std::string imagePath) {
	Inference(Preprocessing(imagePath));

#ifdef IMSHOW
	cv::imshow("charImage", mInputImage);
	cv::waitKey(0);
	cv::destroyWindow("charImage");
#endif

	return GetArgMax();
}

char ASCRecog::GetResult(cv::Mat crop) {
	Inference(Preprocessing(crop));

#ifdef IMSHOW
	cv::imshow("charImage", crop);
	cv::waitKey(0);
	cv::destroyWindow("charImage");
#endif

	return GetArgMax();
}

char ASCRecog::GetArgMax() {
	int result = 0;
	float sum = 0;
	//float max = -std::numeric_limits<float>::max(); // winapi.h ...
	float max = -FLT_MAX;//99999;

	int curOutputSize = mOutputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
	const float* data = mOutputTensors.front().GetTensorMutableData<float>();

	int idx = 0;
	for (int j = 0; j < curOutputSize; j++) {
		sum += data[j];
		if (data[j] > max) {
			max = data[j];
			idx = j;
		}
	}
	result = idx;

	return ASC_MAP[idx];
}