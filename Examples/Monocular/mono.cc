#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

namespace ORB_SLAM3{
    using Seconds = double;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

std::string paddingZeros(const std::string& number, const size_t numberOfZeros = 5){
    std::string zeros{};
    for(size_t iZero{}; iZero < numberOfZeros - number.size(); ++iZero)
        zeros += "0";
    return (zeros + number);
}

void removeSubstring(std::string& str, const std::string& substring) {
    size_t pos;
    while ((pos = str.find(substring)) != std::string::npos) {
        str.erase(pos, substring.length());
    }
}

double ttrack_tot = 0;
int main(int argc, char *argv[])
{
    const int num_seq = 1;

    // ORB_SLAM3  inputs
    string sequence_path;
    string rgb_txt;
    string exp_folder;
    string exp_id{"0"};
    string settings_yaml{"orbslam2_settings.yaml"};
    bool verbose{true};

    string vocabulary{"Vocabulary/ORBvoc.txt"};

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("sequence_path:") != std::string::npos) {
            removeSubstring(arg, "sequence_path:");
            sequence_path =  arg;
            std::cout << "[mono.cc] Path to sequence = " << sequence_path << std::endl;
            continue;
        }
        if (arg.find("rgb_txt:") != std::string::npos) {
            removeSubstring(arg, "rgb_txt:");
            rgb_txt =  arg;
            std::cout << "[mono.cc] Path to rgb_txt = " << rgb_txt << std::endl;
            continue;
        }
        if (arg.find("vocabulary:") != std::string::npos) {
            removeSubstring(arg, "vocabulary:");
            vocabulary = arg;
            std::cout << "[mono.cc] Path to vocabulary = " << vocabulary << std::endl;
            continue;
        }
        if (arg.find("settings_yaml:") != std::string::npos) {
            removeSubstring(arg, "settings_yaml:");
            settings_yaml =  arg;
            std::cout << "[mono.cc] Path to settings_yaml = " << settings_yaml << std::endl;
            continue;
        }
        if (arg.find("exp_folder:") != std::string::npos) {
            removeSubstring(arg, "exp_folder:");
            exp_folder =  arg;
            std::cout << "[mono.cc] Path to exp_folder = " << exp_folder << std::endl;
            continue;
        }
        if (arg.find("exp_id:") != std::string::npos) {
            removeSubstring(arg, "exp_id:");
            exp_id =  arg;
            std::cout << "[mono.cc] Exp id = " << exp_id << std::endl;
            continue;
        }
        if (arg.find("verbose:") != std::string::npos) {
            removeSubstring(arg, "verbose:");
            verbose = bool(std::stoi(arg));
            std::cout << "[mono.cc] Activate Visualization = " << verbose << std::endl;
            continue;
        }
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<ORB_SLAM3::Seconds>> vTimestampsCam;
    vector<int> nImages;
    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "[mono.cc] Loading images for sequence " << seq << "...";
        LoadImages(sequence_path, rgb_txt, vstrImageFilenames[seq], vTimestampsCam[seq]);
        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        if(nImages[seq]<=0)
        {
            cerr << "[mono.cc] ERROR: Failed to load images for sequence" << seq << endl;
            return 1;
        }
        cout << "[mono.cc] LOADED!" << endl;
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabulary, settings_yaml, ORB_SLAM3::System::MONOCULAR, verbose);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    int proccIm=0;
    for (seq = 0; seq<num_seq; seq++)
    {
        // Main loop
        cv::Mat im;
        proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_UNCHANGED); //CV_LOAD_IMAGE_UNCHANGED);
            double tframe = vTimestampsCam[seq][ni];
            if(im.empty())
            {
                cerr << endl << "[mono.cc] Failed to load image at: "
                     <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }
            if(imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
                t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            // Pass the image to the SLAM system
            SLAM.TrackMonocular(im,tframe); 
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            vTimesTrack[ni]=ttrack;
            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];
            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }
        if(seq < num_seq - 1)
        {
            cout << "[mono.cc] Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }
    }
    // Stop all threads
    SLAM.Shutdown();
    // Save camera trajectory
    // string resultsPath_expId = exp_folder + "/" + paddingZeros(exp_id);
    // SLAM.SaveTrajectoryEuRoC(resultsPath_expId + "_" + "CameraTrajectory.txt");
    // SLAM.SaveKeyFrameTrajectoryTUM(resultsPath_expId + "_" + "KeyFrameTrajectory.txt");
    SLAM.SavePointCloud();
    return 0;
}

void LoadImages(const string &pathToSequence, const string &rgb_txt,
    vector<string> &imageFilenames, vector<ORB_SLAM3::Seconds> &timestamps)
{
    ifstream times;
    times.open(rgb_txt.c_str());
    while(!times.eof())
    {
        string s;
        getline(times,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            ORB_SLAM3::Seconds t;
            string sRGB;
            ss >> t;
            timestamps.push_back(t);
            ss >> sRGB;
            imageFilenames.push_back(pathToSequence + "/" +  sRGB);
        }
    }
}
