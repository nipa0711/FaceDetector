#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

const unsigned inWidth = 300;
const unsigned inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

void makeFace(string modelConfiguration, string modelBinary, string source, string output, string fileName)
{
	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
	Mat frame = imread(source);

	if (frame.channels() == 4)
		cvtColor(frame, frame, COLOR_BGRA2BGR);

	if (frame.empty())
		return;
	try {
		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight), meanVal, false, false);
		net.setInput(inputBlob, "data");
		Mat detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		float confidenceThreshold = 0.8;
		int faceCount = 0;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				std::string saveName = fileName;
				Mat face = frame(object);
				if (faceCount > 0) {
					saveName = "";
					saveName = saveName + "multiple_";
					saveName = saveName + to_string(faceCount);
					saveName = saveName + "-" + fileName;
					//saveName.append("multiple_");
					//saveName.append(to_string(faceCount));
					//saveName.append("-" + fileName);
				}
				imwrite(output + "\\" + saveName, face);
				faceCount++;
			}
		}
	}
	catch (Exception e) {

	}
}

int main(int argc, char *argv[])
{
	makeFace(argv[1], argv[2], argv[3], argv[4], argv[5]);
	/*string modelConfiguration, modelBinary, source, output, fileName;
	modelConfiguration = "C:\\Users\\nipa0\\Desktop\\deploy.prototxt";
	modelBinary = "C:\\Users\\nipa0\\Desktop\\res10_300x300_ssd_iter_140000.caffemodel";
	source = "D:\\BlackPink\\Rose\\original\\Rose0010.jpg";
	fileName = "Rose0010.jpg";
	makeFace(modelConfiguration, modelBinary, source, output, fileName);*/
	return 0;
}