#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <map>
#include <regex>
#include <direct.h>

#include "my_logger.h"

#include "logger.h"   // tensorRT 中的头文件
#include "ErrorRecorder.h"
#include "logging.h"

using namespace std;
using namespace my_logger;

//ConsoleLogger my_logger::debug;
//FileLogger my_logger::record("build_at_" __DATE__ "_" __TIME__ ".log");

#ifdef WIN32
#define localtime_r(_Time, _Tm) localtime_s(_Tm, _Time)
#endif
#define localtime_r(_Time, _Tm) localtime_s(_Tm, _Time)

const map<Level, const char *> LevelStr =
{
	{ Level::Debug, "Debug" },
	{ Level::Info, "Info" },
	{ Level::Warning, "Warning" },
	{ Level::Error, "Error" },
	{ Level::Fatal, "Fatal" },
};

ostream& operator<< (ostream& stream, const tm* tm)
{
	return stream << 1900 + tm->tm_year << '-'
		<< setfill('0') << setw(2) << tm->tm_mon + 1 << '-'
		<< setfill('0') << setw(2) << tm->tm_mday << ' '
		<< setfill('0') << setw(2) << tm->tm_hour << ':'
		<< setfill('0') << setw(2) << tm->tm_min << ':'
		<< setfill('0') << setw(2) << tm->tm_sec;
}

BaseLogger::LogStream BaseLogger::operator()(Level nLevel)
{
	return LogStream(*this, nLevel);
}

const tm* BaseLogger::getLocalTime()
{
	auto now = chrono::system_clock::now();
	auto in_time_t = chrono::system_clock::to_time_t(now);
	localtime_r(&in_time_t, &_localTime);
	return &_localTime;
}

void BaseLogger::endline(Level nLevel, string&& oMessage)
{
	_lock.lock();
	output(getLocalTime(), LevelStr.find(nLevel)->second, oMessage.c_str());
	_lock.unlock();
}

void ConsoleLogger::output(const tm *p_tm,
	const char *str_level,
	const char *str_message)
{
	cout << '[' << p_tm << ']'
		<< '[' << str_level << "]"
		<< "\t" << str_message << endl;
	cout.flush();
}

FileLogger::FileLogger(string filename) noexcept
	: BaseLogger()
{
	//string valid_filename(filename.size(), '\0');
	//regex express("/|:| |>|<|\"|\\*|\\?|\\|");
	//regex_replace(valid_filename.begin(),
	//	filename.begin(),
	//	filename.end(),
	//	express,
	//	"_");
	_file.open(filename,
		fstream::out | fstream::app | fstream::ate);
	assert(!_file.fail());
}

FileLogger::~FileLogger()
{
	_file.flush();
	_file.close();
}

void FileLogger::output(const tm *p_tm,
	const char *str_level,
	const char *str_message)
{
	_file << '[' << p_tm << ']'
		<< '[' << str_level << "]"
		<< "\t" << str_message << endl;
	_file.flush();
}


// build和加载引擎的时候需要用到这个
SampleErrorRecorder gRecorder;
namespace sample
{
	Logger gLogger{ Logger::Severity::kINFO };
	LogStreamConsumer gLogVerbose{ LOG_VERBOSE(gLogger) };
	LogStreamConsumer gLogInfo{ LOG_INFO(gLogger) };
	LogStreamConsumer gLogWarning{ LOG_WARN(gLogger) };
	LogStreamConsumer gLogError{ LOG_ERROR(gLogger) };
	LogStreamConsumer gLogFatal{ LOG_FATAL(gLogger) };

	void setReportableSeverity(Logger::Severity severity)
	{
		gLogger.setReportableSeverity(severity);
		gLogVerbose.setReportableSeverity(severity);
		gLogInfo.setReportableSeverity(severity);
		gLogWarning.setReportableSeverity(severity);
		gLogError.setReportableSeverity(severity);
		gLogFatal.setReportableSeverity(severity);
	}
} // namespace sample


namespace my_logger
{
	// \! ------------------------------------日志  Start---------------------------------------

	// \! 判断文件夹是否存在
	// @param: strFolderPath 文件夹路径
	bool DirectoryIfExists(const std::string& strFolderPath)
	{
		if (_access(strFolderPath.c_str(), 0) != -1)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	// \! 递归创建文件夹
	// @param: strFolderPath 多层文件夹路劲
	bool CreateAllDirectory(const std::string& strFolderPath)
	{
		std::string folder_builder;
		std::string sub;
		sub.reserve(strFolderPath.size());
		for (auto it = strFolderPath.begin(); it != strFolderPath.end(); ++it)
		{
			char c = *it;
			sub.push_back(c);
			if (c == '\\' || it == strFolderPath.end() - 1)
			{
				folder_builder.append(sub);
				if (_access(folder_builder.c_str(), 0) == -1)
				{
					if (_mkdir(folder_builder.c_str()) != 0)
					{
						return false;
					}
				}
				sub.clear();
			}
		}

		return true;
	}

	// \! 输出日志
	// @param： params onnx或者caffe的参数结构体
	//			log_str 要输出的日志文字
	void log(const std::string log_path, std::string log_str, Level level) {
		if (log_path != "") {
			bool ret = false;
			if (!DirectoryIfExists(log_path))
			{
				ret = CreateAllDirectory(log_path);
			}
			std::string dst_path = log_path + std::string("\\") + std::string("TRT_LOG.txt");
			my_logger::FileLogger g_log(dst_path);
			g_log(level) << log_str;
		}
		else {
			std::cout << log_str << std::endl;
		}
	}
	// \! -------------------------------------日志 END-----------------------------------------

} // namespace my_logger