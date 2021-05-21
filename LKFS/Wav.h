//
//  Wav.h
//  DSP HW5
//
//  Created by 何冠勳 on 2021/1/29.
//
#ifndef Wav_h
#define Wav_h
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <time.h>
using namespace std;

typedef struct WaveHeader {
protected:
    /*===================Format===================*/
    // riff wave header
    char ChunkID[4] = {'R','I','F','F'};
    unsigned ChunkSize;        // 0 ~ FFFF,FFFF
    char Format[4] = {'W','A','V','E'};
    // fmt subchunk
    char SubChunk1ID[4] = {'f','m','t',' '};
    unsigned SubChunk1Size;    // 0 ~ FFFF,FFFF
    unsigned short AudioFormat;    // 0 ~ FFFF
    unsigned short NumChannels;    // 0 ~ FFFF
    unsigned SampleRate;       // 0 ~ FFFF,FFFF
    unsigned ByteRate;         // 0 ~ FFFF,FFFF
    unsigned short BlockAlign;     // 0 ~ FFFF
    unsigned short BitsPerSample;  // 0 ~ FFFF
    // data subchunk
    char SubChunk2ID[4] = {'d','a','t','a'};
    unsigned SubChunk2Size;    // 0 ~ FFFF,FFFF
    
public:
    /*===================Constructor===================*/
    // Empty Constructor
    WaveHeader() {}
    // Explicit Constructor
    WaveHeader(unsigned short const NC, unsigned const SR, unsigned short const BPS, unsigned NoS) {
        AudioFormat = 1;                  // 1 for PCM...
        SampleRate = SR;
        NumChannels = NC;        // 1 for Mono, 2 for Stereo
        BitsPerSample = BPS;
        ByteRate = (SampleRate * NumChannels * BitsPerSample)/8;
        BlockAlign = (NumChannels * BitsPerSample)/8;
        SubChunk2Size = (NoS * NumChannels * BitsPerSample)/8;
        SubChunk1Size = 16;               // 16 for PCM
        ChunkSize = 4 + (8 + SubChunk1Size) + (8 + SubChunk2Size);
    }
    
    /*===================Destructor===================*/
    //~WaveHeader() { cout << "Waveheader killed" << endl; }
    
    /*===================Get Info===================*/
    unsigned get_ChunkSize() { return ChunkSize; }
    unsigned get_SubChunk1Size() { return SubChunk1Size; }
    unsigned short get_AudioFormat() { return AudioFormat; }
    unsigned short get_NumChannels() { return NumChannels; }
    unsigned get_SampleRate() { return SampleRate; }
    unsigned get_ByteRate() { return ByteRate; }
    unsigned short get_BlockAlign() { return BlockAlign; }
    unsigned short get_BitsPerSample() { return BitsPerSample; }
    unsigned get_SubChunk2Size() { return SubChunk2Size; }
    
    /*===================Clear===================*/
    void clear() {
        ChunkSize = 0; SubChunk1Size = 0; AudioFormat = 0; NumChannels = 0; SampleRate = 0;
        ByteRate = 0; BlockAlign = 0; BitsPerSample = 0; SubChunk2Size = 0;
    }
} WaveHeader;

typedef struct Stereo_Wav {
    /*===================Format===================*/
    WaveHeader header;
    vector<short> left_data;
    vector<short> right_data;

    /*===================Constructor===================*/
    // Empty Constructor
    Stereo_Wav() {}
    // Implicit Constructor by Waveheader
    Stereo_Wav(unsigned short const NC, unsigned const SR, unsigned short const BPS, unsigned NoS):header(NC, SR, BPS, NoS) {}
    // Explicit Constructor
    Stereo_Wav(unsigned short const NC, unsigned const SR, unsigned short const BPS, unsigned NoS, vector<short> ld, vector<short> rd):header(NC, SR, BPS, NoS), left_data(ld), right_data(rd) {}
    
    /*===================Destructor===================*/
    //~Stereo_Wav() { cout << "Stereo_Wav killed" << endl; }
    
    /*===================Clear===================*/
    void clear() {
        header.clear();
        left_data.clear();
        right_data.clear();
    }
    
    /*===================File Operation===================*/
    /*===================Read===================*/
    bool Stereo_WaveRead(string filename) {
        ifstream infile;
        WaveHeader hd;
        vector<short> data;
        
        infile.open(filename, ofstream::binary|ios::in);
        if (!infile.is_open()) {
            cerr << "Could not open the file" << endl;
            return false;
        }
        
        infile.read((char *)&hd, sizeof(hd));
        //cout << header.SampleRate << endl;
        header = hd;

        while(!infile.eof()) {
            short temp;        // data can't be greater than FFFF(65535).
            infile.read((char *)&temp, sizeof(temp));
            data.push_back(temp);
        }
        infile.close();
        
        /*------------------------------------*/
        /* Change data length here for testing*/
        for(unsigned i=0;i<data.size();i=i+2) {
            left_data.push_back(data[i]);
            right_data.push_back(data[i+1]);
        }
        return true;
    }
    /*===================Write===================*/
    bool Stereo_WaveWrite(string filename) {
        ofstream outfile;
        outfile.open(filename, ofstream::binary|ofstream::out);
        if (!outfile.is_open()) {
            cerr << "Could not open the file" << endl;
            return false;
        }
        
        outfile.write((char*)&header, sizeof(header));
        
        for(unsigned i=0;i<left_data.size();i++) {
            outfile.write((char*)&left_data[i], sizeof(left_data[i]));
            outfile.write((char*)&right_data[i], sizeof(right_data[i]));
        }
        outfile.close();
        
        return false;
    }
} Stereo_Wav;

#endif /* Wav_h */
