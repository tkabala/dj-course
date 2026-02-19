package resource_blackouts

import "time"

// ResourceBlackout represents a period when a driver or vehicle is unavailable.
type ResourceBlackout struct {
	ID            int
	DriverID      *int // nullable
	VehicleID     *int // nullable
	BlackoutStart time.Time
	BlackoutEnd   time.Time
	Reason        string
}
