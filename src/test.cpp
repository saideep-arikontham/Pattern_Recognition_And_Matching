/*
  Bruce A. Maxwell
  CS 5330 Spring 2024
  Project 2

  Reads a ResNet18 network and an image and applies the image to the network to generate an embedding

  Also prints out the model layers.  

  The getEmbedding function can be used to get an embedding, or representation of an image
*/
#include <cstdio>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "opencv2/opencv.hpp"  // the top include file
#include "opencv2/dnn.hpp"     // DNN API include file


/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat embedding  holds the embedding vector after the function returns
  cv::dnn::Net net   the pre-trained ResNet 18 network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug=0 );
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug ) {
  const int ORNet_size = 224;
  cv::Mat blob;

  // have the function do the ImageNet mean and SD normalization
  // the function also scales the image to 224 x 224
  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) * (1/0.226), // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  cv::Scalar( 124, 116, 104),  // subtract mean prior to scaling
			  true,   // swapRB
			  false,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!resnetv22_flatten0_reshape0" ); // the name of the embedding layer to grab

  if(debug) {
    cv::imshow( "src", src );
    std::cout << embedding << std::endl;
    std::cout << embedding.rows << " " << embedding.cols << std::endl;
    cv::waitKey(0);
  }

  return(0);
}


/*
  Main function

  Expects two command line arguments: <network path> <image path>

  The image should be a thresholded image with one object in it.
 */
int main(int argc, char *argv[]) {
    char mod_filename[256];
  char img_filename[256];

  if(argc < 2) {
    printf("usage: %s <image path>\n", argv[0] );
    exit(-1);
  }

  strncpy(mod_filename, "./src/resnet18-v2-7.onnx", 255);
  strncpy(img_filename, argv[1], 255);

  // read the network
  cv::dnn::Net net = cv::dnn::readNet( mod_filename );
  printf("Network read successfully\n");

  /// print the names of the layers
  std::vector<cv::String> names = net.getLayerNames();

  for(int i=0;i<names.size();i++) {
    printf("Layer %d: '%s'\n", i, names[i].c_str() );
  }

  // read image 
  cv::Mat src = cv::imread( img_filename );

  // get the embedding
  cv::Mat embedding;
  getEmbedding( src, embedding, net, 1 );  // change the 1 to a 0 to turn off debugging

  return(0);
}
