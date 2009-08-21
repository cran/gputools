getGpuId <- function()
{
	deviceId <- .C("rgetDevice", deviceId = integer(1))$deviceId
	return(deviceId)
}
