#include "plidar/status.h"

const char *plidar_status_str(plidar_status_t s)
{
    switch (s) {
    case PLIDAR_OK:              return "ok";
    case PLIDAR_ERR_ARG:         return "invalid argument";
    case PLIDAR_ERR_DOMAIN:      return "value outside the model domain";
    case PLIDAR_ERR_SATURATED:   return "every cycle detected, inverse undefined";
    case PLIDAR_ERR_NO_CONVERGE: return "iteration did not converge";
    }
    return "unknown";
}
