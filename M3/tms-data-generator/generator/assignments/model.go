package assignments

import "time"

// Assignment represents an actual trip linking an order to a driver and vehicle.
type Assignment struct {
	ID           int
	OrderID      int
	DriverID     int
	VehicleID    int
	BookingStart time.Time
	BookingEnd   time.Time
}
