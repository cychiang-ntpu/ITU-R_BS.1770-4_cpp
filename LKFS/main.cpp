//
//  main.cpp
//  LKFS
//
//  Created by 何冠勳 on 2021/4/19.
//

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "vector_operation.h"
#include "Wav.h"
#include "LKFS.h"
#define pi 3.14159265358979323846
using namespace std;

void peak_normalize(Stereo_Wav &wavein, Stereo_Wav &waveout, double target) {
    vector<double> left_d(wavein.left_data.begin(), wavein.left_data.end());
    vector<double> right_d(wavein.right_data.begin(), wavein.right_data.end());
    scalar(left_d, 1.0/32767.0);   // normalize to [-1,1] by 0xFFFF
    scalar(right_d, 1.0/32767.0);  // normalize to [-1,1] by 0xFFFF
    
    double current_peak = max(*max_element(left_d.begin(), left_d.end()), *max_element(right_d.begin(), right_d.end()));
    double gain = pow(10.0, target/20.0) / current_peak;
    scalar(left_d, gain);
    scalar(right_d, gain);
    
    double normalized_peak = max(*max_element(left_d.begin(), left_d.end()), *max_element(right_d.begin(), right_d.end()));
    if(normalized_peak>=1.0) {
        cout << "Possible clipped samples in output." << endl;
    }
        
    scalar(left_d, 32767.0);   // normalize back to [0~FFFF]
    scalar(right_d, 32767.0);  // normalize back to [0~FFFF]
    vector<short> left_s(left_d.begin(), left_d.end());
    vector<short> right_s(right_d.begin(), right_d.end());
    waveout.left_data.clear(); waveout.left_data = left_s;
    waveout.right_data.clear(); waveout.left_data = right_s;
}

void loudness_normalize(Stereo_Wav &wavein, Stereo_Wav &waveout, double target_loudness, double input_loudness) {
    vector<double> left_d(wavein.left_data.begin(), wavein.left_data.end());
    vector<double> right_d(wavein.right_data.begin(), wavein.right_data.end());
    scalar(left_d, 1.0/32767.0);   // normalize to [-1,1] by 0xFFFF
    scalar(right_d, 1.0/32767.0);  // normalize to [-1,1] by 0xFFFF
    
    double delta = target_loudness - input_loudness;
    double gain = pow(10.0, delta/20.0);
    scalar(left_d, gain);
    scalar(right_d, gain);
    
    double normalized_peak = max(*max_element(left_d.begin(), left_d.end()), *max_element(right_d.begin(), right_d.end()));
    if(normalized_peak>=1.0) {
        cout << "Possible clipped samples in output." << endl;
    }
    
    scalar(left_d, 32767.0);   // normalize back to [0~FFFF]
    scalar(right_d, 32767.0);  // normalize back to [0~FFFF]
    vector<short> left_s(left_d.begin(), left_d.end());
    vector<short> right_s(right_d.begin(), right_d.end());
    waveout.left_data.clear(); waveout.left_data = left_s;
    waveout.right_data.clear(); waveout.left_data = right_s;
}

// -i -o argc argv
int main() {
    string infname;
    infname = "input.wav";
    Stereo_Wav wavein, waveout;
    wavein.Stereo_WaveRead(infname);
    
    double loudness = 0.0;
    
    time_t start, end;
    time(&start);
    
    loudness = integrated_loudness(wavein, wavein.header.get_SampleRate());
    cout << "Loudness : " << setprecision(17) << loudness << endl;
    
    time(&end);
    cout << "(spent " << double(end-start) << " seconds to compute)" << endl;
    
    //peak_normalize(wavein, waveout, -12.0);
    loudness_normalize(wavein, waveout, -12.0, loudness);
    
    wavein.clear();
    
    /*
    string outfname = "test.wav";
    Stereo_Wav waveout(2, 8000, 16, wavein.left_data.size(), wavein.left_data, wavein.right_data);
    waveout.Stereo_WaveWrite(outfname);*/
    
    /*
    char keyin;
    cout << "compute LKFS with your file : press key '1' " << endl;
    cout << "test LKFS by input.wav : press key '2'" << endl;
    cout << "to end this program : press key '3'" << endl;
    cout << "press a key : ";
    cin >> keyin;
    
    string infname;
    Stereo_Wav wavein;
    bool proceed = true;

    while(proceed) {
        switch (keyin) {
            case '1':
                cout << "type your wav filename : " << endl;
                cin >> infname;
                WaveRead(infname, wavein);
                LKFS(wavein);
                wavein.clear();
                break;
            case '2':
                infname = "input.wav";
                WaveRead(infname, wavein);
                LKFS(wavein);
                wavein.clear();
            case '3':
                proceed = false;
                break;
            default:
                break;
        }
    }*/
}
