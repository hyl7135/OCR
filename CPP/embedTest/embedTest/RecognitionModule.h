#pragma once

#include "ASCRecog.h"
#include "HanCharRecog.h"
#include "TextDetector.h"

class RecognitionModule {
private:
#ifdef LOG
	const std::string mLogPath{ "./" };
	const std::string mLoggerName{ "logger" };
	std::string mLogFileName{};

	std::shared_ptr<spdlog::logger> mLogger;
#endif

	ASCRecog mASC_recog;
	HanCharRecog mHAN_recog;
	TextDetector mText_detect;

public:
#ifdef LOG
	//RecognitionModule(std::string path = "./", std::string name = "testLog") {
	RecognitionModule() {
		std::string path = "./";
		std::string name = "testLog";
		mLogFileName = name;
		mLogFileName = path + mLogFileName + GetTimeString() + ".log";
		mLogger = spdlog::basic_logger_mt(mLoggerName, mLogFileName);
	}
#else
	RecognitionModule() = default;
#endif

	void moduleTest(std::string imagePath);
};