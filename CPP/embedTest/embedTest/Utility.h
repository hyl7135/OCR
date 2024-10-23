#pragma once

#include <filesystem>
#include <vector>

#include <Windows.h>

#include "resource.h"

#include "opencv2/opencv.hpp"
#include "spdlog/sinks/basic_file_sink.h"

//ifdef MSVC
#ifndef LINUX
#pragma warning(disable:4996)
#endif // !LINUX



HMODULE GCM();

std::string GetTimeString();

std::pair<std::vector<UCHAR>, DWORD> GetEmbedData(int first, int second);

std::vector<std::string> GetImageFileFromDir(std::string path);

std::vector<std::string> GetImageFilesFromPath(std::string path);

void PaddingTEST(std::string path);

cv::Mat ImagePadding(const cv::Mat&, int minLen);

cv::Mat ResizeKeepingRatio(const cv::Mat& src, int pivot);


// LOG 
namespace std
{
	inline std::string to_string(std::string text) { return text; }
	inline std::string to_string(const char* text) { std::string rtn = text; return rtn; };
}

inline int GetLen(const char* str) { return strlen(str); }
inline int GetLen(const std::string& str) { return str.size(); }
inline int GetLen(int digit) { int cnt = 0; while (digit = digit / 10) cnt++; return cnt + 1; }
inline int GetLen(long digit) { int cnt = 0; while (digit = digit / 10) cnt++; return cnt + 1; }

template <typename strType, typename... Strings>
int GetLen(const strType& baseStr, Strings... args)
{
	return GetLen(baseStr) + GetLen(args...);
}

inline void Appending(std::string& concat_str) { return; }

//template <typename... Strings>
//void Appending(std::string& baseStr, long digit, Strings... strs)
//{
//	baseStr.append(std::to_string(digit));
//	Appending(baseStr, strs...);
//}

template <typename String, typename... Strings>
void Appending(std::string& concat_str, String curStr, Strings... strs)
{
	concat_str.append(std::to_string(curStr));
	Appending(concat_str, strs...);
}

template <typename strType, typename... Strings>
std::string StrConcat(const strType& baseStr, Strings... strs)
{
	std::string rtnStr;
	rtnStr.reserve(GetLen(baseStr, strs...));

	rtnStr.append(baseStr);
	Appending(rtnStr, strs...);

	return rtnStr;
}


inline void WriteLog_Debug(std::shared_ptr<spdlog::logger> logger, const std::string& msg) { if (logger != nullptr) logger->debug(msg); }
inline void WriteLog_Info(std::shared_ptr<spdlog::logger> logger, const std::string& msg) { if (logger != nullptr) logger->info(msg); }
inline void WriteLog_Error(std::shared_ptr<spdlog::logger> logger, const std::string& msg) { if (logger != nullptr) logger->error(msg); }
inline void WriteLog_Warn(std::shared_ptr<spdlog::logger> logger, const std::string& msg) { if (logger != nullptr) logger->warn(msg); }
inline void WriteLog_Critical(std::shared_ptr<spdlog::logger> logger, const std::string& msg) { if (logger != nullptr) logger->critical(msg); }

#define MAKE_LOG_FORMAT(...) (StrConcat("\t FUNCTION : ", __FUNCTION__, " line(", std::to_string(__LINE__), ") -> ", ##__VA_ARGS__))

#define LOG_DEBUG(logger, ...)		(WriteLog_Debug(logger, MAKE_LOG_FORMAT(##__VA_ARGS__)))
#define LOG_INFO(logger, ...)		(WriteLog_Info(logger, MAKE_LOG_FORMAT(##__VA_ARGS__)))
#define LOG_WARN(logger, ...)		(WriteLog_Warn(logger, MAKE_LOG_FORMAT(##__VA_ARGS__)))
#define LOG_ERROR(logger, ...)		(WriteLog_Error(logger, MAKE_LOG_FORMAT(##__VA_ARGS__)))
#define LOG_CRITICAL(logger, ...)	(WriteLog_Critical(logger, MAKE_LOG_FORMAT(##__VA_ARGS__)))