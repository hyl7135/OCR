#include "RecognitionModule.h"

void RecognitionModule::moduleTest(std::string imagePath) {
	auto seg = mText_detect.GetResult(imagePath);
	//auto orgImg = mText_detect.GetOrgImg().clone();
	const auto& orgImg = mText_detect.GetOrgImg();
	auto inputImg = mText_detect.GetInputImage();
	auto outputSize = mText_detect.GetOutputSize();

#ifdef LOG
	LOG_INFO(mLogger, "Image File : ", imagePath);
#endif

	//std::max(inputImg.rows, inputImg.rows);
	float imageRatio = MAX(inputImg.rows, inputImg.rows) / MAX(outputSize.width, outputSize.height);
	int idx = 0;

	cv::Size unPaddingSize;

	if (orgImg.cols > orgImg.rows) {
		float ratio = orgImg.rows / (float)orgImg.cols;

		unPaddingSize.width = outputSize.width;
		unPaddingSize.height = ratio * outputSize.height;
	}
	else {
		float ratio = orgImg.cols / (float)orgImg.rows;

		unPaddingSize.width = ratio * outputSize.width;
		unPaddingSize.height = outputSize.height;
	}

	float widthRatio = inputImg.cols / (float)outputSize.width;
	float heightRatio = inputImg.cols / (float)outputSize.height;

	for (auto& [sentenceRect, charRects] : seg) {
		float widthRatio = sentenceRect.x / (float)outputSize.width;

		auto drawRect = sentenceRect;

		drawRect.x = orgImg.cols * (sentenceRect.x / (float)unPaddingSize.width);
		drawRect.y = orgImg.rows * (sentenceRect.y / (float)unPaddingSize.height);
		drawRect.width = orgImg.cols * (sentenceRect.width / (float)unPaddingSize.width);
		drawRect.height = orgImg.rows * (sentenceRect.height / (float)unPaddingSize.height);

#ifdef LOG
		LOG_INFO(mLogger, "Sentence Coor : ", drawRect.x, ", ", drawRect.y, ", ", drawRect.width, ", ", drawRect.height);
#endif

		//cv::Scalar colorPalN = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
		//cv::rectangle(testImg, drawRect, cv::Scalar(0, 0, 255), 1, 8);
		//cv::imshow("Sentence", testImg(drawRect).clone());
		std::cout << "Sentence " << ++idx << std::endl;
		for (auto& charRect : charRects) {
			int padding = 1;
			charRect.x = orgImg.cols * (charRect.x / (float)unPaddingSize.width) - padding;
			charRect.y = orgImg.rows * (charRect.y / (float)unPaddingSize.height) - padding;
			charRect.width = orgImg.cols * (charRect.width / (float)unPaddingSize.width) + padding * 2;
			charRect.height = orgImg.rows * (charRect.height / (float)unPaddingSize.height) + padding * 2;

#ifdef LOG
			LOG_INFO(mLogger, "Char Coor : ", charRect.x, ", ", charRect.y, ", ", charRect.width, ", ", charRect.height);
#endif

			cv::Mat charImg = orgImg(charRect).clone();

			auto asc = mASC_recog.GetResult(charImg);
			//if (asc == ASC_MAP[sizeof(ASC_MAP) + 1]) {
			if (asc == NOT_ASCII) {
				std::wcout << mHAN_recog.GetResult(charImg) << " ";
			}
			else {
				std::cout << asc << " ";
			}


#ifdef IMSHOW
			cv::imshow("crop", charImg);
			cv::waitKey(0);
			cv::destroyWindow("crop");
#endif
		}
		std::cout << "\n";
	}
}
