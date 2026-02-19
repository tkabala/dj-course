package driver_shifts

// DriverShift represents a recurring shift schedule for a driver.
type DriverShift struct {
	ID        int
	DriverID  int
	DayOfWeek int    // 0 = Sunday, 6 = Saturday
	StartTime string // HH:MM:SS format
	EndTime   string // HH:MM:SS format
}
