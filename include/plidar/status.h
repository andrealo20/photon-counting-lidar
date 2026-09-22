/**
 * @file status.h
 * @brief Return codes shared by every module.
 *
 * Nothing in this library allocates and nothing aborts. A function that
 * cannot honour its contract says so through a status code and leaves the
 * caller's buffers untouched, so a simulation sweep can report a bad
 * parameter combination instead of dying halfway through a run.
 */
#ifndef PLIDAR_STATUS_H
#define PLIDAR_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PLIDAR_OK = 0,
    PLIDAR_ERR_ARG,       /**< null pointer, zero size or parameter out of domain */
    PLIDAR_ERR_DOMAIN,    /**< the model is not defined for these values */
    PLIDAR_ERR_SATURATED, /**< every cycle produced a detection, the inverse is undefined */
    PLIDAR_ERR_NO_CONVERGE
} plidar_status_t;

const char *plidar_status_str(plidar_status_t s);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_STATUS_H */
