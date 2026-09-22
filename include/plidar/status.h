/**
 * @file status.h
 * @brief Return codes shared by every module.
 *
 * Nothing in this library allocates and nothing aborts. A function that
 * cannot honour its contract says so through a status code, so a sweep can
 * report a bad parameter combination and carry on instead of dying halfway
 * through a run.
 *
 * A status other than PLIDAR_OK says nothing about the output buffer. Some
 * failures are found while checking arguments, before anything is written;
 * others are found part way through, with earlier entries already filled in.
 * plidar_coates_invert is the clearest case: it discovers that the cycles
 * have run out only when it reaches the bin where they do. Treat the output
 * of a failed call as undefined rather than as unchanged.
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
