// -------------------------------------------------------------------------------------------
// Saideep Arikontham
// Feb 2025
// Pattern Recognition and Computer Vision - Project 2
// -------------------------------------------------------------------------------------------


#include <cstdlib>
#include <dirent.h>
#include <cstdio> // gives printf
#include <cstring> // gives strcpy
#include <opencv2/opencv.hpp> // openCV
#include "csv_util.h" // CSV utility functions
#include "utils.h" //Utilify functions for feature vectors


int main(int argc, char *argv[]) {
    // check for a command line argument
    if(argc < 3 ) {
        printf("usage: %s <image dirname> <feature set>\n", argv[0]); // argv[0] is the program name
        exit(-1);
    }

    char dir_name[256];
    strncpy(dir_name, argv[1], 255); // directory name

    char feature_set[256];
    strncpy(feature_set, argv[2], 255); // feature set name

    //Comparing Feature sets
    if (strcmp(feature_set, "7x7_middle_square") == 0){ // Using 7x7 Middle square for computing histogram feature vectors
        printf("Using 7x7 feature set\n");

        char feature_file[256];
        strncpy(feature_file, "./feature_vectors/feature_vectors_7x7.csv", 255);
        write_feature_vectors(dir_name, feature_file, get_feature_vector_7x7);
    }

    else if (strcmp(feature_set, "rg_histogram") == 0){ // Using rg chromacity histogram to get feature vectors
        printf("Using RG histogram set\n");

        char feature_file[256];
        strncpy(feature_file, "./feature_vectors/feature_vectors_rg_hist.csv", 255);
        write_feature_vectors(dir_name, feature_file, get_feature_vector_rg_hist);
    }

    else if (strcmp(feature_set, "two_rgb_histogram") == 0){ // Using two rgb histograms , one full and the other is a middle block of the image for feature vectors
        printf("Using two RGB histogram set\n");

        char feature_file[256];
        strncpy(feature_file, "./feature_vectors/feature_vectors_two_rgb_hist.csv", 255);
        write_feature_vectors(dir_name, feature_file, get_feature_vector_two_rgb_hist);
    }

    else if (strcmp(feature_set, "rgb_texture_hist") == 0){ // Using one rgb histogram and one magnitude texture histogram for feature vectors
        printf("Using RGB histogram and Texture Histogram set\n");

        char feature_file[256];
        strncpy(feature_file, "./feature_vectors/feature_vectors_rgb_texture_hist.csv", 255);
        write_feature_vectors(dir_name, feature_file, get_feature_vector_rgb_texture_hist);
    }

    else if (strcmp(feature_set, "depth_rgb") == 0){ // Using one rgb histogram with depth thresholding
        printf("Using depth RGB histogram \n");

        char feature_file[256];
        strncpy(feature_file, "./feature_vectors/feature_vectors_depth_rgb_hist.csv", 255);
        write_feature_vectors(dir_name, feature_file, get_feature_vector_depth_rgb_hist);
    }

    else if (strcmp(feature_set, "dnn") == 0){ // Using deep neural network vectors
        printf("Using deep neural network vectors\n");

        char feature_file[256];
        strncpy(feature_file, "./feature_vectors/feature_vectors_dnn.csv", 255);
        write_feature_vectors(dir_name, feature_file, get_feature_vector_dnn);
    }

    else{
        printf("Unknown feature set %s\n", feature_set);
        exit(-1);
    }
}
