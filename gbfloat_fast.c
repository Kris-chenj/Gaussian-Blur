#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <omp.h> 
//inplement dymanic

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159

typedef struct FVec
{
    unsigned int length;
    unsigned int min_length;
    unsigned int min_deta;
    float* data;
    float* sum;
} FVec;

typedef struct Image
{
    unsigned int dimX, dimY, numChannels;
    float* data;
} Image;

void normalize_FVec(FVec v)
{
    // float sum = 0.0;
    unsigned int i,j;
    int ext = v.length / 2;
    v.sum[0] = v.data[ext];
    for (i = ext+1,j=1; i < v.length/4; i+=4,j+=4)
    {
        v.sum[j] = v.sum[j-1] + v.data[i]*2;
        v.sum[j+1] = v.sum[j] + v.data[i+1]*2;
        v.sum[j+2] = v.sum[j+1] + v.data[i+2]*2;
        v.sum[j+3] = v.sum[j+2] + v.data[i+3]*2;
    }
    for(;i < v.length;++i,++j){v.sum[j] = v.sum[j-1] + v.data[i]*2;}
    // float ttow = 2.0;
    // __m128 multip = _mm_setr_ps(ttow,ttow,ttow,ttow);
    // #pragma omp parallel for private(i,j,v,ttow,multip)
    // for (i = ext+1,j=1; i < v.length/4*4; i+=4,j+=4)
    // {
    //     __m128 arr_sum_cal = _mm_loadu_ps(&(v.sum[j-1]));
    //     __m128 arr_data_cal = _mm_loadu_ps(&(v.sum[i]));
    //     arr_data_cal = _mm_mul_ps(arr_data_cal,multip);
    //     __m128 arr_sum_des =_mm_add_ps(arr_sum_cal,arr_data_cal);
    //     _mm_storeu_ps(&(v.sum[j]),arr_sum_des);
    //     // v.sum[j] = v.sum[j-1] + v.data[i]*2;
    // }
    // for(i = v.length/4*4; i < v.length; ++i)
    // {v.sum[j] = v.sum[j-1] + v.data[i]*2;++j;}

    // for (i = 0; i <= ext; i++)
    // {
    //      v.data[i] /= v.sum[v.length - ext - 1 ] ;
    //      printf("%lf ",v.sum[i]);
    // }
}

float* get_pixel(Image img, int x, int y)
{
    if (x < 0)
    {
        x = 0;
    }
    if (x >= img.dimX)
    {
        x = img.dimX - 1;
    }
    if (y < 0)
    {
        y = 0;
    }
    if (y >= img.dimY)
    {
        y = img.dimY - 1;
    }
    return img.data + img.numChannels * (y * img.dimX + x);
}

float gd_origin(float a, float b, float x)
{
    float c = (x-b) / a;
    return exp((-.5) * c * c) / (a * sqrt(2 * PI));
}

FVec make_gv(float a, float x0, float x1, unsigned int length, unsigned int min_length)
{
    FVec v;
    v.length = length;
    v.min_length = min_length;
    if(v.min_length > v.length){
        v.min_deta = 0;
    }else{
        v.min_deta = ((v.length - v.min_length) / 2);
    }
    v.data = malloc(length * sizeof(float));
    v.sum = malloc((length / 2 + 1)* sizeof(float));
    float step = (x1 - x0) / ((float)length);
    int offset = length/2;

    #pragma omp parallel for firstprivate(a,length,offset,step)
    for (int i = 0; i < length/4; i+=4)
    {
        v.data[i] = gd_origin(a, 0.0f, (i-offset)*step);
        v.data[i+1] = gd_origin(a, 0.0f, (i+1-offset)*step);
        v.data[i+2] = gd_origin(a, 0.0f, (i+2-offset)*step);
        v.data[i+3] = gd_origin(a, 0.0f, (i+3-offset)*step);
    }
    for(int i = length/4;i<length;++i){v.data[i] = gd_origin(a, 0.0f, (i-offset)*step);}
    // #pragma omp parallel for firstprivate(a,length,offset,step)
    // for (int i = 0; i < length/4; i+=4)
    // {
    //     __m128 aa = _mm_set1_ps(a);
    //     __m128 bb = _mm_set1_ps(0.0f);
    //     __m128 cc = _mm_set1_ps(offset);
    //     __m128 dd = _mm_set1_ps(step);
    //     __m128 ii = _mm_setr_ps((float)i,(float)i+1,(float)i+2,(float)i+3);
    //     __m128 cal = _mm_mul_ps(_mm_sub_ps(ii,cc),dd);
    //     _mm_storeu_ps(&v.data[i],gd(aa,bb,cal));
    //     // v.data[i] = gd(a, 0.0f, (i-offset)*step);
    // }
    for (int i = length/4; i < length; i++){v.data[i] = gd_origin(a, 0.0f, (i-offset)*step);}
    normalize_FVec(v);
    return v;
}

void print_fvec(FVec v)
{
    unsigned int i;
    printf("\n");
    for (i = 0; i < v.length; i++)
    {
        printf("%f ", v.data[i]);
    }
    printf("\n");
}

Image img_sc(Image a)
{
    Image b = a;
    b.data = malloc(b.dimX * b.dimY * b.numChannels * sizeof(float));
    return b;
}

Image trans(Image a){
    Image b;
    b.dimX=a.dimY;
    b.dimY=a.dimX;
    b.numChannels=a.numChannels;
    b.data=(float*)malloc(sizeof(float)*a.dimX*a.dimY*3);
    for(int x = 0;x < b.dimX;x++){
        for(int y = 0;y < b.dimY;y++){
            memcpy(b.data+(x+y*b.dimX)*3,a.data+(y+x*a.dimX)*3,sizeof(float)*3);
        }
    }
    return b;
}


Image gb_h(Image a, FVec gv)
{
    Image b = img_sc(a);

    int ext = gv.length / 2;
    int offset;
    unsigned int x, y, channel;
    float *pc;
    Image new_image;
    new_image.numChannels = 3;
    new_image.dimX = a.dimX + 2*ext;
    new_image.dimY = a.dimY;
    new_image.data = (float*)malloc(new_image.dimX*new_image.dimY*3*sizeof(float));
    float* gvdata;
    gvdata = (float*)malloc(3*gv.length*sizeof(float));
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < gv.length; s++)
    {
        memcpy(gvdata+s*3,gv.data+s,sizeof(float));
        memcpy(gvdata+s*3+1,gv.data+s,sizeof(float));
        memcpy(gvdata+s*3+2,gv.data+s,sizeof(float));
    }
    
    #pragma omp parallel for schedule(dynamic)
    for(int m = 0;m<new_image.dimY;++m)
    {
        memcpy(new_image.data + (ext + m*new_image.dimX)*3 , a.data + m*a.dimX*3 , sizeof(float) * 3 *(a.dimX));
        for(int n = 0;n<ext;++n)
        {
            memcpy(new_image.data + (n + m*new_image.dimX)*3 , a.data + m*a.dimX*3 , sizeof(float) * 3);
            memcpy(new_image.data + (n + a.dimX + ext + m*new_image.dimX)*3 , a.data + (m*a.dimX + a.dimX - 1)*3, sizeof(float) * 3);
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (y = 0; y < a.dimY; y++)
    {
        for (x = 0; x < a.dimX; x++)
        {
            float* pix;
            int i;
            __m256 weight;
            __m256 temp;
            // pc = get_pixel(b, x, y);
            //pc = new_image.data + 3 * (y * new_image.dimX + x + ext);
            pc = b.data + 3 * (y * b.dimX + x);

            unsigned int deta = fmin(fmin(a.dimY-y-1, y),fmin(a.dimX-x-1, x));
            deta = fmin(deta, gv.min_deta);
            __m128 arr = _mm_setzero_ps();
            __m256 arr1 = _mm256_setzero_ps();
            __m256 arr2 = _mm256_setzero_ps();
            __m256 arr3 = _mm256_setzero_ps();
            for (i = deta; i < (gv.length - deta) - 8; i += 8)
            {
                // offset = i - ext;
                //float dummy = gv.data[i]/gv.sum[ext - deta];
                weight = _mm256_loadu_ps(gvdata + i*3);
                // pix = get_pixel(a, x + offset, y);
                pix = new_image.data + 3 * (y * new_image.dimX + x + i);
                // pix = new_image.data + 3 * (y * new_image.dimX + x + ext + offset);
                temp = _mm256_loadu_ps(pix);
                arr1 = _mm256_add_ps(arr1,(_mm256_mul_ps(temp,weight)));
                // sum += gv.data[i]/gv.sum[ext - deta] * (float)get_pixel(a, x + offset, y)[channel];


                // offset = i - ext;
                //float dummy = gv.data[i]/gv.sum[ext - deta];
                weight = _mm256_loadu_ps(gvdata + i*3 + 8);
                // pix = get_pixel(a, x + offset, y);
                //pix = new_image.data + 3 * (y * new_image.dimX + x + i + );
                pix += 8;
                // pix = new_image.data + 3 * (y * new_image.dimX + x + ext + offset);
                temp = _mm256_loadu_ps(pix);
                arr2 = _mm256_add_ps(arr2,(_mm256_mul_ps(temp,weight)));
                // sum += gv.data[i]/gv.sum[ext - deta] * (float)get_pixel(a, x + offset, y)[channel];


                // offset = i - ext;
                //float dummy = gv.data[i]/gv.sum[ext - deta];
                weight = _mm256_loadu_ps(gvdata + i*3 + 16);
                // pix = get_pixel(a, x + offset, y);
                //pix = new_image.data + 3 * (y * new_image.dimX + x + i + );
                pix += 8;
                // pix = new_image.data + 3 * (y * new_image.dimX + x + ext + offset);
                temp = _mm256_loadu_ps(pix);
                arr3 = _mm256_add_ps(arr3,(_mm256_mul_ps(temp,weight)));
                // sum += gv.data[i]/gv.sum[ext - deta] * (float)get_pixel(a, x + offset, y)[channel];

            }
            arr[0] = arr1[0] + arr1[3] + arr1[6] + arr2[1] + arr2[4] + arr2[7] + arr3[2] + arr3[5];
            arr[1] = arr1[1] + arr1[4] + arr1[7] + arr2[2] + arr2[5] + arr3[0] + arr3[3] + arr3[6];
            arr[2] = arr1[2] + arr1[5] + arr2[0] + arr2[3] + arr2[6] + arr3[1] + arr3[4] + arr3[7];
            for(;i < gv.length - deta;++i)
            {
                //float dummy = gv.data[i]/gv.sum[ext - deta];
                __m128 weigh = _mm_set1_ps(gv.data[i]); 
                // pix = get_pixel(a, x + offset, y);
                pix = new_image.data + 3 * (y * new_image.dimX + x + i);
                __m128 tem = _mm_loadu_ps(pix);
                arr = _mm_add_ps(arr,(_mm_mul_ps(tem,weigh)));
            }
            //_mm_storeu_ps(tem,arr);
            pc[0] = arr[0] / gv.sum[ext - deta];
            pc[1] = arr[1] / gv.sum[ext - deta];
            pc[2] = arr[2] / gv.sum[ext - deta];
            // pc[channel] = sum; 
        }
    }
    free(gvdata);
    free(new_image.data);
    return b;
}

Image apply_gb(Image a, FVec gv)
{
    Image b = gb_h(a, gv);
    Image c = trans(b);
    Image d = gb_h(c, gv);
    Image e = trans(d);
    free(b.data);
    free(c.data);
    free(d.data);

    return e;
}

int main(int argc, char** argv)
{
    struct timeval start_time, stop_time, elapsed_time; 
    gettimeofday(&start_time,NULL);
    if (argc < 6)
    {
        printf("Usage: ./gb.exe <inputjpg> <outputname> <float: a> <float: x0> <float: x1> <unsigned int: dim>\n");
        exit(0);
    }

    float a, x0, x1;
    unsigned int dim, min_dim;

    sscanf(argv[3], "%f", &a);
    sscanf(argv[4], "%f", &x0);
    sscanf(argv[5], "%f", &x1);
    sscanf(argv[6], "%u", &dim);
    sscanf(argv[7], "%u", &min_dim);

    FVec v = make_gv(a, x0, x1, dim, min_dim);
    // print_fvec(v);
    Image img;
    img.data = stbi_loadf(argv[1], &(img.dimX), &(img.dimY), &(img.numChannels), 0);

    Image imgOut = apply_gb(img, v);
    stbi_write_jpg(argv[2], imgOut.dimX, imgOut.dimY, imgOut.numChannels, imgOut.data, 90);
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); 
    printf("%f \n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    free(imgOut.data);
    free(v.data);
    free(v.sum);
    return 0;
}