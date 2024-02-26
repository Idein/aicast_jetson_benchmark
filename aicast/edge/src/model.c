#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <math.h>
#include "hailo/hailort.h"

#define MAX_EDGE_LAYERS (16)
#define HEF_FILE ("yolox_s_leaky.hef")
static hailo_vdevice vdevice = NULL;
static hailo_hef hef = NULL;
static hailo_configure_params_t config_params = {0};
static hailo_configured_network_group network_group = NULL;
static size_t network_group_size = 1;
static hailo_input_vstream_params_by_name_t input_vstream_params[MAX_EDGE_LAYERS] = {0};
static hailo_output_vstream_params_by_name_t output_vstream_params[MAX_EDGE_LAYERS] = {0};
static hailo_vstream_info_t output_vstreams_info[MAX_EDGE_LAYERS] = {0};
static hailo_input_vstream input_vstreams[MAX_EDGE_LAYERS] = {NULL};
static hailo_output_vstream output_vstreams[MAX_EDGE_LAYERS] = {NULL};
static size_t input_vstreams_size = MAX_EDGE_LAYERS;
static size_t output_vstreams_size = MAX_EDGE_LAYERS;

#define READ_AND_DEQUANTIZE(idx, type, out, size) \
    do { \
        type buf[size]; \
        hailo_status status = HAILO_UNINITIALIZED; \
        status = hailo_vstream_read_raw_buffer(output_vstreams[idx], buf, size * sizeof(type)); \
        assert(status == HAILO_SUCCESS); \
        float scale = output_vstreams_info[idx].quant_info.qp_scale; \
        float zp = output_vstreams_info[idx].quant_info.qp_zp; \
        for (int i = 0; i < size; i++) \
            out[i] = scale * (buf[i] - zp); \
    } while (0)

typedef struct {
    unsigned char *input;
    float *output0;
    float *output1;
    float *output2;
    int N;
} thread_args_t;

void *write_thread_func(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    for(int i = 0; i < args-> N; i++) {
        const int input_size = 640 * 640 * 3;
        hailo_status status = hailo_vstream_write_raw_buffer(input_vstreams[0], args->input, input_size);
        args->input += input_size;
        assert(status == HAILO_SUCCESS);
    }
    return NULL;
}

void *read_thread_func(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    for(int i = 0; i < args-> N; i++) {
        const int output_size0 = 80 * 80 * 85;
        const int output_size1 = 40 * 40 * 85;
        const int output_size2 = 20 * 20 * 85;

        READ_AND_DEQUANTIZE(0, uint8_t, args->output0, output_size0);
        READ_AND_DEQUANTIZE(1, uint8_t, args->output1, output_size1);
        READ_AND_DEQUANTIZE(2, uint8_t, args->output2, output_size2);
        args->output0 += output_size0;
        args->output1 += output_size1;
        args->output2 += output_size2;
    }
    return NULL;
}


void infer_thread(unsigned char *input0, float *out0, float *out1, float *out2, int N) 
{
    thread_args_t arg;
    pthread_t write_thread;
    pthread_t read_thread;

    arg.input = input0;
    arg.output0 = out0;
    arg.output1 = out1;
    arg.output2 = out2;
    arg.N = N;
    pthread_create(&write_thread, NULL, write_thread_func, &arg);
    pthread_create(&read_thread, NULL, read_thread_func, &arg);

    pthread_join(write_thread, NULL);
    pthread_join(read_thread, NULL);
}

void infer(unsigned char *input0, float *out0, float *out1, float *out2)
{
    hailo_status status = HAILO_UNINITIALIZED;
    status = hailo_vstream_write_raw_buffer(input_vstreams[0], input0, 640 * 640 * 3);
    assert(status == HAILO_SUCCESS);
    status = hailo_flush_input_vstream(input_vstreams[0]);
    assert(status == HAILO_SUCCESS);

    READ_AND_DEQUANTIZE(0, uint8_t, out0, 80*80*85);
    READ_AND_DEQUANTIZE(1, uint8_t, out1, 40*40*85);
    READ_AND_DEQUANTIZE(2, uint8_t, out2, 20*20*85);
}

int init()
{
    hailo_status status = HAILO_UNINITIALIZED;
    hailo_vdevice_params_t params = {0};
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.device_count = 1;
    status = hailo_create_vdevice(&params, &vdevice);

    assert(status == HAILO_SUCCESS);

    status = hailo_create_hef_file(&hef, HEF_FILE);
    assert(status == HAILO_SUCCESS);

    status = hailo_init_configure_params(hef, HAILO_STREAM_INTERFACE_PCIE, &config_params);
    assert(status == HAILO_SUCCESS);

    status = hailo_configure_vdevice(vdevice, hef, &config_params, &network_group, &network_group_size);
    assert(status == HAILO_SUCCESS);

    status = hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
                                             input_vstream_params, &input_vstreams_size);
    assert(status == HAILO_SUCCESS);

    status = hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
                                              output_vstream_params, &output_vstreams_size);
    assert(status == HAILO_SUCCESS);

    status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstreams_size, input_vstreams);
    assert(status == HAILO_SUCCESS);

    status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstreams_size, output_vstreams);
    assert(status == HAILO_SUCCESS);

    for (size_t i = 0; i < output_vstreams_size; i++)
        hailo_get_output_vstream_info(output_vstreams[i], &output_vstreams_info[i]);

    return status;
}

void destroy()
{
    (void)hailo_release_output_vstreams(output_vstreams, output_vstreams_size);
    (void)hailo_release_input_vstreams(input_vstreams, input_vstreams_size);
    (void)hailo_release_hef(hef);
    (void)hailo_release_vdevice(vdevice);
}