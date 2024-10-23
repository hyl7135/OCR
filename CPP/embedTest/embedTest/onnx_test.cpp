#include "onnx_test.h"

//#include "TextDetector.h"

TextDetector_tmp::TextDetector_tmp(std::string modelPath) {
	// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/shared_lib/test_inference.cc
	//modelPath = mModelPath;
	mModelPath = modelPath;

	auto tmpEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TextDetector");
	mEnv = std::move(tmpEnv);

	//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(mSessionOptions, 0));
	//// 추가 graph 최적화 옵션 
	//mSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//OrtCUDAProviderOptionsV2* cuda_options = nullptr;
	//Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));


	// GPU ?
	//OrtApi::Update
	//std::vector<const char*> keys{ "device_id", "gpu_mem_limit", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace", "cudnn_conv1d_pad_to_nc1d" };
	//std::vector<const char*> values{ "0", "2147483648", "kSameAsRequested", "DEFAULT", "1", "1", "1" };
	//Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size()));

	//mSessionOptions.AppendExecutionProvider_CUDA_V2(*cuda_options);

	auto embedData = GetEmbedData(DETECTION, DETECTION_FILE);
	auto curPath = std::filesystem::current_path();

	
	
	std::string curPathStr = curPath.string();
	modelPath = curPathStr + "\\" + modelPath;

	std::cout << "curPath : " << curPathStr << std::endl;
	std::cout << "FullPath : " << modelPath << std::endl;

	std::wstring wideStr;
	wideStr.assign(modelPath.begin(), modelPath.end());

	std::wcout << "f : " << wideStr << std::endl;

	//Ort::Session a();
	//auto tmpSession = std::make_unique<Ort::Session>(*mEnv, wideStr.c_str(), mSessionOptions);
	//auto tmpSession = std::make_unique<Ort::Session>(*mEnv, modelPath.c_str(), mSessionOptions);

	auto tmpSession = std::make_unique<Ort::Session>(*mEnv, &(embedData.first[0]), embedData.second, mSessionOptions);

	mSession = std::move(tmpSession);

	const size_t num_input_nodes = (*mSession).GetInputCount();
	std::vector<Ort::AllocatedStringPtr> input_names_ptr;
	input_names_ptr.reserve(num_input_nodes);
	mInputNodeNames.reserve(num_input_nodes);

	for (size_t i = 0; i < num_input_nodes; i++) {
		// print input node names
		auto input_name = (*mSession).GetInputNameAllocated(i, mAllocator);
		std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
		mInputNodeNames.push_back(input_name.get());
		input_names_ptr.push_back(std::move(input_name));

		// print input node types
		auto type_info = (*mSession).GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		std::cout << "Input " << i << " : type = " << type << std::endl;

		// print input shapes/dims
		mInputNodeDims = tensor_info.GetShape();
		std::cout << "Input " << i << " : num_dims = " << mInputNodeDims.size() << '\n';
		for (size_t j = 0; j < mInputNodeDims.size(); j++) {
			std::cout << "Input " << i << " : dim[" << j << "] =" << mInputNodeDims[j] << '\n';
		}
		std::cout << std::flush;

		for (const auto& i : mInputNodeNames) {
			std::cout << "nodeSize : " << mInputNodeNames.size() << std::endl;
			std::cout << "NodeName : " << i << std::endl;
		}
	}
}


void TextDetector_tmp::Inference(std::string imagePath) {
	mOrgImage = cv::imread(imagePath);

	mInputImage = ImagePadding(mOrgImage, kInputImageSize.width);

	cv::imshow("Org", mInputImage);

	//mInputImage = static_resize(mInputImage, kInputImageSize);
	//cv::Mat blob = cv::dnn::blobFromImage(mInputImage, 1 / 255.0, kInputImageSize, (0, 0, 0), true, false);
	cv::Mat blob = cv::dnn::blobFromImage(mInputImage, 1 / 255.0, cv::Size(768, 768), (0, 0, 0), true, false);
	constexpr size_t input_tensor_size = 768 * 768 * 3;

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size,
		mInputNodeDims.data(), mInputNodeDims.size());

	assert(input_tensor.IsTensor());

	for (const auto& i : mInputNodeNames) {
		std::cout << "NodeSize : " << mInputNodeNames.size() << std::endl;
		std::cout << "NodeName : " << i << std::endl;
	}

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	mOutputTensors =
		(*mSession).Run(Ort::RunOptions{ nullptr }, kTmpInputNodeNames.data(), &input_tensor, 1, mDetectOutputNodeNames.data(), 1);
	//(*mSession).Run(Ort::RunOptions{ nullptr }, mInputNodeNames.data(), &input_tensor, 1, mDetectOutputNodeNames.data(), 1);
	std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
	std::cout << "Taken time to inference : " << sec.count() << std::endl;

	assert(mOutputTensors.size() == 1 && mOutputTensors.front().IsTensor());
}

void TextDetector_tmp::ImageProcessing3() {
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
			std::cout << "Diff : " << diff[i] << " ";
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

				std::cout << "Pixel : " << pixel[i] << ", Thresh * 1.5 : " << PIXEL_THRESH << std::endl;
				if (pixel[i] > PIXEL_THRESH) continue;
				//if (abs(diff[i]) < 0.001) continue;

				std::cout << "Inflection : " << diff[i] << std::endl;
				//xPtrs.push_back(i);
				xPtrs.push_back(i + boundingBox.x + 1);
				cv::circle(testImg, cv::Point(i + boundingBox.x, boundingBox.y), 3, cv::Scalar(0, 0, 255), 1);
			}
		}
		xPtrs.push_back(boundingBox.br().x);
		std::cout << "\n\n";

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

	cv::imshow("TEST img", testImg);
	cv::imshow("SegMat", segMat);
	cv::imshow("TextMask", textMaskImage);
	cv::imshow("LinkMask", linkMaskImage);
	cv::imshow("SetenceMask", sentenceMask);
	cv::waitKey(0);
}

void TextDetector_tmp::ImageProcessing2() {
	const float* outputDataRaw = mOutputTensors.front().GetTensorMutableData<float>();
	int dataNum = mOutputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

	mInferenceROI.clear();

	cv::Mat textMaskImage(kOutputImageSize, CV_32FC1);
	cv::Mat linkMaskImage(kOutputImageSize, CV_32FC1);

	memcpy(textMaskImage.data, outputDataRaw, dataNum / 2 * sizeof(float));
	memcpy(linkMaskImage.data, outputDataRaw + dataNum / 2, dataNum / 2 * sizeof(float));

	cv::threshold(textMaskImage, textMaskImage, mTextThreshold, 1, 0);
	cv::threshold(linkMaskImage, linkMaskImage, mLinkThreshold, 1, 0);

	cv::Mat linkMaskImageC1;
	linkMaskImage.convertTo(linkMaskImageC1, CV_8UC1);
	linkMaskImageC1 *= 255;

	cv::Mat sentenceMask = textMaskImage | linkMaskImage;
	//cv::threshold(sentenceMask, sentenceMask, 0.4, 1, 0);
	sentenceMask *= 255;
	sentenceMask.convertTo(sentenceMask, CV_8UC1);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(sentenceMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat testImg = cv::Mat::zeros(textMaskImage.cols, textMaskImage.cols, CV_8UC3);
	std::vector<std::vector<int>> innerPtrs;

	int pivot = 0;

	for (const auto& points : contours) {
		auto boundingBox = cv::boundingRect(points);
		int padding = 1;
		boundingBox.x -= padding;
		boundingBox.y -= padding;
		boundingBox.height += padding * 2;
		boundingBox.width += padding * 2;


		cv::rectangle(testImg, cv::boundingRect(points), cv::Scalar(255, 255, 255), 1, 8, 0);

		std::vector<std::vector<cv::Point>> linkContours;
		std::vector<cv::Vec4i> linkHierarchy;

		cv::findContours(linkMaskImageC1(boundingBox), linkContours, linkHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		std::vector<int> storage;

		for (const auto& innerPoints : linkContours) {
			//std::cout << "Storage Size : " << storage.size() << std::endl;
			int sumX = 0;

			int rx = (1 << (sizeof(int) * 8 - 1));
			int lx = ~rx;
			auto charBoundingBox = cv::boundingRect(innerPoints);

			/*cv::Scalar colorPalN = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
			cv::rectangle(testImg, charBoundingBox + cv::Point(boundingBox.tl()), colorPalN, 1, 8);*/

			storage.emplace_back((charBoundingBox.x + charBoundingBox.br().x) / 2);
		}

		for (auto& pt : storage) {
			pt += boundingBox.x;
		}

		storage.push_back(boundingBox.tl().x);
		storage.push_back(boundingBox.br().x);

		auto aSort = [](int post, int back) -> bool {return post < back; };
		std::sort(storage.begin(), storage.end(), aSort);

		std::pair<cv::Rect, std::vector<cv::Rect>> tmpRois;
		tmpRois.first = boundingBox;

		int ty = boundingBox.y;
		int by = boundingBox.br().y;

		////////////////////////////////////////////////////////////////////
		std::vector<cv::Rect> charRects;
		for (int i = 0; i < storage.size() - 1; i++) {
			cv::Rect tmpRect(cv::Point(storage[i], ty), cv::Point(storage[i + 1], by));
			charRects.push_back(tmpRect);
		}
		tmpRois.second = std::move(charRects);
		mInferenceROI.push_back(tmpRois);
		////////////////////////////////////////////////////////////////////

		//std::cout << "InnerPtrs Size : " << innerPtrs.size() << std::endl;
		//for (int i = 0; i < innerPtrs.size(); i++) {
		//	auto& vecRef = innerPtrs[i];
		//	std::vector<cv::Rect> charRects;

		//	std::cout << "Character Num : " << vecRef.size() << std::endl;
		//	for (int j = 0; j < vecRef.size() - 1; j++) {
		//		cv::Rect tmpRect(cv::Point(vecRef[j], ty), cv::Point(vecRef[j + 1], by));
		//		charRects.push_back(tmpRect);

		//		cv::rectangle(testImg, tmpRect, cv::Scalar(0,0,255), 1, 8);
		//		cv::imshow("testImg", testImg);
		//		cv::waitKey(0);
		//	}
		//	tmpRois.second = std::move(charRects);
		//}
		//mInferenceROI.push_back(tmpRois);

		//innerPtrs.push_back(std::move(storage));

		//cv::Scalar colorPal = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

		//for (const auto& i : innerPtrs[pivot]) {
		//	cv::circle(testImg, cv::Point(i, boundingBox.y), 3, colorPal, 1);
		//	//cv::imshow("TestImg", testImg);
		//	//cv::waitKey(0);
		//}

		////////////////////////////////////////////////////////////////////

		++pivot;
	}

	cv::Scalar colorPal = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
	for (const auto& [sentenceRect, charRectVec] : mInferenceROI) {
		cv::rectangle(testImg, sentenceRect, colorPal, 1);
		for (const auto& charRect : charRectVec) {
			cv::rectangle(testImg, charRect, colorPal, 1);
		}
	}

	cv::imshow("Text", textMaskImage);
	cv::imshow("Link", linkMaskImage);
	cv::imshow("sentenceMask", sentenceMask);
	cv::imshow("testImg", testImg);
	cv::waitKey(0);
}

void TextDetector_tmp::ROI_Test2() {
	//cv::Mat testImg = mOrgImage.clone();

	//float imageRatio = std::max(mInputImage.rows, mInputImage.rows) / std::max(kOutputImageSize.width, kOutputImageSize.height);
	//int idx = 0;

	//cv::Size unPaddingSize;
	//if (mOrgImage.cols > mOrgImage.rows) {
	//	float ratio = mOrgImage.rows / (float)mOrgImage.cols;

	//	unPaddingSize.width = kOutputImageSize.width;
	//	unPaddingSize.height = ratio * kOutputImageSize.height;
	//}
	//else {
	//	float ratio = mOrgImage.cols / (float)mOrgImage.rows;

	//	unPaddingSize.width = ratio * kOutputImageSize.width;
	//	unPaddingSize.height = kOutputImageSize.height;
	//}

	//std::cout << "unPaddingSize : " << unPaddingSize << std::endl;

	//float widthRatio = mInputImage.cols / (float)kOutputImageSize.width;
	//float heightRatio = mInputImage.rows / (float)kOutputImageSize.height;

	////float widthRatio = (float)kOutputImageSize.width / mInputImage.cols;
	////float heightRatio = (float)kOutputImageSize.height / mInputImage.rows;

	//CharacterRecognizer recog("model\\model_8_2.onnx");
	//for (auto& [sentenceRect, charRects] : mInferenceROI) {
	//	float widthRatio = sentenceRect.x / (float)kOutputImageSize.width;

	//	auto drawRect = sentenceRect;

	//	drawRect.x = mOrgImage.cols * (sentenceRect.x / (float)unPaddingSize.width);
	//	drawRect.y = mOrgImage.rows * (sentenceRect.y / (float)unPaddingSize.height);
	//	drawRect.width = mOrgImage.cols * (sentenceRect.width / (float)unPaddingSize.width);
	//	drawRect.height = mOrgImage.rows * (sentenceRect.height / (float)unPaddingSize.height);

	//	cv::Scalar colorPalN = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
	//	//cv::rectangle(testImg, drawRect, cv::Scalar(0, 0, 255), 1, 8);
	//	cv::imshow("Sentence", testImg(drawRect).clone());

	//	std::cout << "Sentence " << ++idx << std::endl;
	//	for (auto& charRect : charRects) {
	//		int padding = 1;
	//		charRect.x = mOrgImage.cols * (charRect.x / (float)unPaddingSize.width) - padding;
	//		charRect.y = mOrgImage.rows * (charRect.y / (float)unPaddingSize.height) - padding;
	//		charRect.width = mOrgImage.cols * (charRect.width / (float)unPaddingSize.width) + padding * 2;
	//		charRect.height = mOrgImage.rows * (charRect.height / (float)unPaddingSize.height) + padding * 2;

	//		recog.Inference(testImg(charRect).clone());
	//		recog.ImageProcessing();
	//	}
	//	std::cout << "\n";
	//}
}

void TextDetector_tmp::ROI_Test() {
	//cv::Mat testImg = mInputImage.clone();
	//CharacterRecognizer recog("");
	//int idx = 0;

	//for (const auto& [sentenceRect, charRects] : mInferenceROI) {
	//	auto drawRect = sentenceRect;
	//	drawRect.x *= 2;
	//	drawRect.y *= 2;
	//	drawRect.height *= 2;
	//	drawRect.width *= 2;

	//	cv::Scalar colorPalN = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
	//	cv::imshow("Sentence", testImg(drawRect).clone());
	//	//cv::rectangle(testImg, drawRect, colorPalN, 3, 8);

	//	std::cout << "Sentence " << ++idx << std::endl;
	//	for (const auto& charRect : charRects) {
	//		auto drawCharRect = charRect;
	//		drawCharRect.x *= 2;
	//		drawCharRect.y *= 2;
	//		drawCharRect.height *= 2;
	//		drawCharRect.width *= 2;

	//		recog.Inference(testImg(drawCharRect).clone());
	//		recog.ImageProcessing();/*

	//		cv::rectangle(testImg, drawCharRect, colorPalN, 1, 8);
	//		cv::imshow("Roi Test", testImg);
	//		cv::waitKey(0);*/
	//	}
	//	std::cout << "\n";

	//}
	//// std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> mInferenceROI; 

	//cv::imshow("Roi Test", testImg);
	//cv::waitKey(0);
}

cv::Mat TextDetector_tmp::GetRoiFromOrg() {

}

void TextDetector_tmp::ImageProcessing() {
	const float* outputDataRaw = mOutputTensors.front().GetTensorMutableData<float>();
	int dataNum = mOutputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

	cv::Mat textMaskImage(kOutputImageSize, CV_32FC1);
	cv::Mat linkMaskImage(kOutputImageSize, CV_32FC1);

	memcpy(textMaskImage.data, outputDataRaw, dataNum / 2 * sizeof(float));
	memcpy(linkMaskImage.data, outputDataRaw + dataNum / 2, dataNum / 2 * sizeof(float));

	cv::Mat textMap, linkMap;
	cv::threshold(textMaskImage, textMap, mTextThreshold, 1, 0);
	cv::threshold(linkMaskImage, linkMap, mLinkThreshold, 1, 0);
	cv::imshow("TextMapBe", textMap);

	cv::Mat Merged;
	Merged = textMaskImage.clone();
	Merged.convertTo(Merged, CV_8UC1, 255);
	cv::threshold(Merged, Merged, mLinkThreshold * 255, 255, 0);

	cv::Mat comp, stats, centroids;
	//int LabelNum = cv::connectedComponentsWithStats(Merged, comp, stats, centroids, 4);
	textMap = textMap - linkMap;
	textMap.convertTo(textMap, CV_8UC1, 255);
	auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
	cv::erode(textMap, textMap, kernel, cv::Point(-1, -1), 3);
	cv::imshow("TextMap", textMap);
	int LabelNum = cv::connectedComponentsWithStats(textMap, comp, stats, centroids, 4);

	cv::Mat copyImage = mInputImage.clone();
	cv::resize(copyImage, copyImage, cv::Size(384, 384));

	float WidthRatioInAndOut;
	float HeightRatioInAndOut;

	int widthRate = 1;//copyImage.cols / 368;
	int heightRate = 1;// copyImage.rows / 368;


	float niter = 0;
	for (int i = 1; i < LabelNum; i++) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		int x = stats.at<int>(i, cv::CC_STAT_LEFT);
		int y = stats.at<int>(i, cv::CC_STAT_TOP);
		int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

		int pad = 2;

		x = MAX(x - pad, 0);
		y = MAX(y - pad, 0);
		width += pad * 2;
		height += pad * 2;


		//if (area < 5) {
		//	continue;
		//}

		std::cout << "mWidthRatio : " << mWidthRatio << ", " << mHeightRatio << std::endl;
		mCoors.push_back(cv::Rect((float)x / kOutputImageSize.width * mInputImage.cols, (float)y / kOutputImageSize.height * mInputImage.rows,
			(float)width / kOutputImageSize.width * mInputImage.cols, (float)height / kOutputImageSize.height * mInputImage.rows));
		//mCoors.push_back(cv::Rect((float)x / mWidthRatio, (float)y / mHeightRatio,
		//	(float)width / mWidthRatio, (float)height / mWidthRatio));

		niter = sqrt(area * MIN(width, height) / (width * height) * 2);

		auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));

		int sx = x - niter;
		int ex = x + width + niter + 1;
		int sy = y - niter;
		int ey = y + height + niter + 1;

		sx = MAX(0, sx);
		sy = MAX(0, sy);
		ex = MIN(Merged.cols, ex);
		ey = MIN(Merged.rows, ey);

		std::cout << sx << ", " << sy << ", " << ex << ", " << ey << std::endl;

		//cv::Rect roiRect = cv::Rect(sx, sy, ex, ey);
		cv::Rect roiRect = cv::Rect(cv::Point(sx, sy), cv::Point(ex, ey));
		cv::Mat roi = Merged(roiRect);

		cv::dilate(roi, roi, kernel);
	}



	/*cv::imshow("TEXT1", text1);
	cv::imshow("TEXT2", text2);*/

	cv::imshow("textMap", textMap);
	cv::imshow("linkMap", linkMap);
	cv::imshow("Merged", Merged);

	test();
	cv::waitKey(0);
}

void TextDetector_tmp::test() {
	cv::Mat testImage = mInputImage.clone();

	for (const auto& i : mCoors) {
		std::cout << "Rect : " << i << std::endl;
		cv::rectangle(testImage, i, cv::Scalar(255, 0, 0), 1, 8);
	}

	cv::imshow("testImage", testImage);
}

cv::Mat TextDetector_tmp::PassRoi(int idx) {
	if (idx < 0 || idx > mCoors.size()) {
		std::cerr << "Invalid Index" << std::endl;
		//exit(0);
	}

	//cv::Mat ROI = mInputImage(mCoors[idx]);

	cv::Mat ROI = mInputImage(mCoors[idx]).clone();
	cv::cvtColor(ROI, ROI, cv::COLOR_BGR2GRAY);
	ROI += 130;
	cv::threshold(ROI, ROI, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
	cv::cvtColor(ROI, ROI, cv::COLOR_GRAY2BGR);

	cv::copyMakeBorder(ROI, ROI, 4, 4, 4, 4, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

	return ROI;
}

cv::Mat TextDetector_tmp::RegularDoc(int idx) {
	if (idx < 0 || idx > mCoors.size()) {
		std::cerr << "Invalid Index" << std::endl;
		//exit(0);
	}

	//cv::Mat ROI = mInputImage(mCoors[idx]);

	cv::Mat ROI = mInputImage(mCoors[idx]).clone();

	cv::copyMakeBorder(ROI, ROI, 4, 4, 4, 4, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

	return ROI;
}


int TextDetector_tmp::GetTextNum() {
	return mCoors.size();
}