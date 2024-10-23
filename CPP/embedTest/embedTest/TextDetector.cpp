#include "TextDetector.h"

cv::Mat TextDetector::Preprocessing(std::string imagePath) {
	mOrgImage = cv::imread(imagePath);
	mInputImage = ImagePadding(mOrgImage, kInputImageSize.width);

	cv::Mat blob = cv::dnn::blobFromImage(mInputImage, 1 / 255.0, kInputImageSize, (0, 0, 0), true, false);
	return blob;
}

void TextDetector::Test(std::string imagePath) {
	Inference(Preprocessing(imagePath));
	//GetResult();
}

std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> TextDetector::GetResult(std::string imagePath) {
	Inference(Preprocessing(imagePath));
	ImageProcessing();

	return mInferenceROI;
}


void TextDetector::ImageProcessing() {
	const float* outputDataRaw = mOutputTensors.front().GetTensorMutableData<float>();
	int dataNum = mOutputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

	mInferenceROI.clear();

	cv::Mat textMaskImage(kOutputImageSize, CV_32FC1);
	cv::Mat linkMaskImage(kOutputImageSize, CV_32FC1);

	memcpy(textMaskImage.data, outputDataRaw, dataNum / 2 * sizeof(float));
	memcpy(linkMaskImage.data, outputDataRaw + dataNum / 2, dataNum / 2 * sizeof(float));

	cv::Mat segMat = textMaskImage.clone();

	cv::threshold(textMaskImage, textMaskImage, mTextThreshold, 1, 0);
	cv::threshold(linkMaskImage, linkMaskImage, mLinkThreshold, 1, 0);

	cv::Mat sentenceMask = textMaskImage | linkMaskImage;
	//cv::threshold(sentenceMask, sentenceMask, std::max(mTextThreshold, mLinkThreshold), 1, 0);
	sentenceMask *= 255;
	sentenceMask.convertTo(sentenceMask, CV_8UC1);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(sentenceMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat testImg = cv::Mat::zeros(textMaskImage.cols, textMaskImage.cols, CV_8UC3);
	std::vector<std::vector<int>> innerPtrs;
	// std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> mInferenceROI; 

	int pivot = 0;

	for (const auto& points : contours) {
		auto boundingBox = cv::boundingRect(points);
		int padding = 2;
		boundingBox.x -= padding;
		boundingBox.y -= padding;
		boundingBox.height += padding * 2;
		boundingBox.width += padding * 2;

		float* ptr = (float*)segMat.data
			+ segMat.rows * ((boundingBox.y + boundingBox.br().y) / 2)
			+ boundingBox.x;

		float lastVal = 0;
		//int direction = ASC;

		std::vector<float> diff;
		diff.reserve(boundingBox.width - 1);

		for (int i = 0; i < boundingBox.width; i++) {
			float* curPixel = ptr + i;
			//diff[i] = *curPixel - *(curPixel + 1);
			diff.push_back(*curPixel - *(curPixel + 1));
			//std::cout << "Diff : " << diff[i] << " ";
		}

		std::vector<int> xPtrs;
		cv::rectangle(testImg, boundingBox, cv::Scalar(255, 255, 255), 1, 8, 0);

		std::pair<cv::Rect, std::vector<cv::Rect>> tmpRois;
		std::vector<cv::Rect> charVec;
		float* pixel = ptr;

		xPtrs.push_back(boundingBox.x);
		for (int i = 0; i < diff.size() - 1; i++) {
			if (diff[i] * diff[i + 1] < 0) {

				float PIXEL_THRESH = 0.6;

				//std::cout << "Pixel : " << pixel[i] << ", Thresh * 1.5 : " << PIXEL_THRESH << std::endl;
				if (pixel[i] > PIXEL_THRESH) continue;
				//if (abs(diff[i]) < 0.001) continue;

				//std::cout << "Inflection : " << diff[i] << std::endl;
				//xPtrs.push_back(i);
				xPtrs.push_back(i + boundingBox.x + 1);
				cv::circle(testImg, cv::Point(i + boundingBox.x, boundingBox.y), 3, cv::Scalar(0, 0, 255), 1);
			}
		}
		xPtrs.push_back(boundingBox.br().x);
		//std::cout << "\n\n";

		std::vector<cv::Rect> tmpRects;
		for (int i = 0; i < xPtrs.size() - 1; i++) {
			int padding = 1;

			tmpRects.push_back(
				cv::Rect(
					cv::Point(xPtrs[i] - padding, boundingBox.y - padding),
					cv::Point(xPtrs[i + 1] + padding, boundingBox.br().y + padding)));

			cv::rectangle(testImg, tmpRects[i], cv::Scalar(255, 0, 0), 1, 8, 0);
		}
		std::pair<cv::Rect, std::vector<cv::Rect>> tmpPairs;
		tmpPairs.first = boundingBox;
		tmpPairs.second = std::move(tmpRects);

		mInferenceROI.push_back(std::move(tmpPairs));

		//std::vector<cv::Rect> charRects;
		//for (int i = 0; i < storage.size() - 1; i++) {
		//	cv::Rect tmpRect(cv::Point(storage[i], ty), cv::Point(storage[i + 1], by));
		//	charRects.push_back(tmpRect);
		//}
	}
	std::reverse(mInferenceROI.begin(), mInferenceROI.end());
	//std::sort(mInferenceROI.rbegin(), mInferenceROI.rend());

#ifdef IMSHOW
	cv::imshow("ORG", mInputImage);
	cv::imshow("TEST img", testImg);
	cv::imshow("SegMat", segMat);
	cv::imshow("TextMask", textMaskImage);
	cv::imshow("LinkMask", linkMaskImage);
	cv::imshow("SetenceMask", sentenceMask);
	cv::waitKey(0);
#endif
}