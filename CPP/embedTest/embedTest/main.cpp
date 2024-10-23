#include <iostream>
#include <vector>
#include <filesystem>

#include <Windows.h>	

#include "resource.h"
#include "onnx_test.h"
#include "RecognitionModule.h"

#include "cxxopts.hpp"

int ASCII_test(std::string imagePath) {
	ASCRecog a;

	std::vector<std::string> fileVec = GetImageFilesFromPath(imagePath);
	for (const auto& i : fileVec) {
		std::cout << "Image : " << i << std::endl;
		std::cout << "Char : " << a.GetResult(i) << std::endl;
	}
}

int Han_test(std::string imagePath) {
	HanCharRecog a;

	std::vector<std::string> fileVec = GetImageFilesFromPath(imagePath);
	for (const auto& i : fileVec) {
		std::cout << "Image : " << i << std::endl;
		std::wcout << "Char : " << a.GetResult(i) << std::endl;
	}
}

int Mix_test(std::string imagePath) {
	ASCRecog ascRecog;
	HanCharRecog hanRecog;

	std::vector<std::string> fileVec = GetImageFilesFromPath(imagePath);
	for (const auto& file : fileVec) {
		cv::Mat imgMat = cv::imread(file);

		auto asc = ascRecog.GetResult(imgMat);
		if (asc == NOT_ASCII) {
			std::wcout << hanRecog.GetResult(imgMat) << std::endl;
		}
		else {
			std::cout << asc << std::endl;
		}
	}
}

int module_test(std::string imagePath) {
	RecognitionModule module;

	std::vector<std::string> fileVec = GetImageFilesFromPath(imagePath);
	for (const auto& i : fileVec) {
		std::cout << "File : " << i << std::endl;
		module.moduleTest(i);
	}

	return 0;
}

void TestWithOpts(int argc, char** argv) {
	cxxopts::Options options("teset", "test description");

	if (argc > 1) {
		options.allow_unrecognised_options().add_options()
			//("h,help", "How to use")
			("h,hanguel", "hanguel character Image", cxxopts::value<bool>()->default_value("false"))
			("m,mix", "recognize hanguel and ascii character", cxxopts::value<bool>()->default_value("false"))
			("t,text", "multi character image", cxxopts::value<bool>()->default_value("false"))
			("imagePath", "imagePath", cxxopts::value<std::string>())
			;
	}
	else {
		return;
	}

	auto opt = options.parse(argc, argv);
	bool charType = 0;
	bool mixMode = 0;
	bool mode = 0;
	std::string imagePath;

	if (opt.count("imagePath")) {
		imagePath = opt["imagePath"].as<std::string>();
	}

	if (opt.count("text")) {
		mode = opt["text"].as<bool>();
	}

	if (opt.count("mix")) {
		std::cout << "mix" << std::endl;
		mixMode = opt["mix"].as<bool>();
	}

	if (opt.count("h")) {
		std::cout << " hangeul !" << std::endl;;
		std::cout << "Hanguel opt : " << opt["h"].as<bool>() << std::endl;
		charType = opt["h"].as<bool>();
	}

	std::cout << "Image Path : " << imagePath << std::endl;

	if (mode) {
		module_test(imagePath);
	}
	else {
		if (mixMode) {
			Mix_test(imagePath);
		}
		else {
			if (charType) {
				Han_test(imagePath);
			}
			else {
				ASCII_test(imagePath);
			}
		}
	}

}

// https://github.com/microsoft/onnxruntime/issues/6475
// https://www.google.com/search?q=c%2B%2B+onnxruntime+load+model+with+data&sca_esv=7bfb32b140944815&sca_upv=1&ei=MsbzZpzQDPnn2roPk-vNgQs&ved=0ahUKEwicse6X092IAxX5s1YBHZN1M7AQ4dUDCBA&uact=5&oq=c%2B%2B+onnxruntime+load+model+with+data&gs_lp=Egxnd3Mtd2l6LXNlcnAiJGMrKyBvbm54cnVudGltZSBsb2FkIG1vZGVsIHdpdGggZGF0YTIFECEYoAEyBRAhGKABSOE1ULIpWLE0cAF4AJABAJgBkQGgAakJqgEEMS4xMLgBA8gBAPgBAZgCBqACkwTCAgoQABiwAxjWBBhHwgIGEAAYCBgewgIIEAAYgAQYogSYAwCIBgGQBgqSBwMyLjSgB7QY&sclient=gws-wiz-serp

// --imagePath D:\project\private\TextRecognizer\SingleChar\0
// -h --imagePath D:\project\private\TextRecognizer\SingleChar\0
// -m --imagePath D:\project\private\TextRecognizer\SingleChar\0
// -t --imagePath D:\project\private\TextRecognizer\multiText\sample


int main(int argc, char** argv) {
	//EmbdingTest();
	std::cout << Ort::GetVersionString() << std::endl;

	TestWithOpts(argc, argv);

	//SegTest2("D:\\project\\private\\TextRecognizer\\multiText\\sample");

	//text_test("D:\\project\\private\\TextRecognizer\\multiText\\sample");
	// 
	//Han_test("D:\\project\\private\\TextRecognizer\\SingleChar\\0");
	//ASCII_test("D:\\project\\private\\TextRecognizer\\SingleChar\\0");

	//module_test("D:\\project\\private\\TextRecognizer\\multiText\\sample\\image.jpg");
}