#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

const int inWidth = 300;
const int inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

string modelConfiguration;
string modelBinary;
string saveDir;
string type;
dnn::Net net;

void makeFace(Mat frame, string fileName);

void init(string prototxt, string caffeModel, string output, string videoOrimage)
{
	modelConfiguration = prototxt;
	modelBinary = caffeModel;
	net = readNetFromCaffe(modelConfiguration, modelBinary);
	saveDir = output;
	type = videoOrimage;
}

void loadFromImage(string source, string fileName)
{
	Mat frame = imread(source);

	if (frame.empty())
		return;

	if (frame.channels() == 4)
		cvtColor(frame, frame, COLOR_BGRA2BGR);

	makeFace(frame, fileName);
}

void loadFromVideo(string source, string fileName)
{
	VideoCapture capture(source);
	if (!capture.isOpened())
	{
		cout << "please check video file" << endl;
	}

	Mat frame;
	while (true)
	{
		capture >> frame;
		if (frame.empty())
		{
			cout << "frame is empty" << endl;
			return;
		}

		makeFace(frame, fileName);

		if (waitKey(1) >= 0)
			break;
	}
}

void saveFaceFile(Mat face, string fileName, int faceCount)
{
	string saveName = fileName;

	if (faceCount > 0) {
		saveName = "";
		saveName.append("multiple_");
		saveName.append(to_string(faceCount));
		saveName.append("-" + fileName);
	}
	imwrite(saveDir + "\\" + saveName, face);
}

void makeFace(Mat frame, string fileName)
{
	try {
		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight), meanVal, false, false);
		net.setInput(inputBlob, "data");
		Mat detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		inputBlob.release();
		detection.release();

		double confidenceThreshold = 0.8;
		int faceCount = 0;

		register int xLeftBottom;
		register int yLeftBottom;
		register int xRightTop;
		register int yRightTop;
		register float confidence;
		Mat face;

		for (register int i = 0; i < detectionMat.rows; i++)
		{
			confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				face = frame(object);
				if (type.compare("image") == 0)
				{
					saveFaceFile(face, fileName, faceCount);
				}
				else
				{
					saveFaceFile(face, fileName, faceCount);
					//rectangle(frame, object, Scalar(0, 255, 0));
					//resizeWindow("result", 1280, 720);
					//imshow("result", frame);
				}
				faceCount++;
			}
		}
		face.release();
	}
	catch (Exception e) {

	}
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		cout << "Wrong input" << endl;
		return 0;
	}
	// modelConfiguration, modelBinary, source, output, fileName, type

	string type = argv[6];
	init(argv[1], argv[2], argv[4], type);
	if (type.compare("image") == 0)
	{
		loadFromImage(argv[3], argv[5]);
	}
	else
	{
		loadFromVideo(argv[3], argv[5]);
	}
	/*
	modelConfiguration = "C:\\Users\\nipa0\\Desktop\\deploy.prototxt";
	modelBinary = "C:\\Users\\nipa0\\Desktop\\res10_300x300_ssd_iter_140000.caffemodel";
	source = "D:\\BlackPink\\Rose\\original\\Rose0010.jpg";
	fileName = "Rose0010.jpg";
	makeFace(modelConfiguration, modelBinary, source, output, fileName);*/
	return 0;
}