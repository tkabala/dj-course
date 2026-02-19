package resource_blackouts

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"tms-data-generator/generator/drivers"
	"tms-data-generator/generator/vehicles"
)

var driverReasons = []string{"Holiday", "Sick Leave", "Personal Day", "Training", "Jury Duty"}
var vehicleReasons = []string{"Scheduled Maintenance", "Repair", "Inspection", "Tire Replacement", "Recall Service"}

// GenerateResourceBlackouts generates blackout periods for drivers and vehicles.
// Each driver/vehicle gets at most one ongoing blackout and non-overlapping historical ones.
// activeDriverIDs and activeVehicleIDs list resources currently on an active assignment — they are excluded from ongoing blackouts.
func GenerateResourceBlackouts(count int, driversList []drivers.Driver, vehiclesList []vehicles.Vehicle, activeDriverIDs map[int]bool, activeVehicleIDs map[int]bool) []ResourceBlackout {
	blackouts := make([]ResourceBlackout, 0, count)
	now := time.Now()

	// Track the end of the last blackout per driver/vehicle to prevent overlaps
	driverEnd := make(map[int]time.Time)
	vehicleEnd := make(map[int]time.Time)
	// Track which drivers/vehicles already have an ongoing blackout
	driverOngoing := make(map[int]bool)
	vehicleOngoing := make(map[int]bool)

	for i := 1; i <= count; i++ {
		var driverID *int
		var vehicleID *int
		var reason string

		// Roughly 60% driver blackouts, 40% vehicle blackouts
		isDriver := rand.Float64() < 0.6
		if isDriver {
			id := driversList[rand.Intn(len(driversList))].ID
			driverID = &id
			reason = driverReasons[rand.Intn(len(driverReasons))]
		} else {
			id := vehiclesList[rand.Intn(len(vehiclesList))].ID
			vehicleID = &id
			reason = vehicleReasons[rand.Intn(len(vehicleReasons))]
		}

		// Determine the earliest this blackout can start (after the last one ends)
		var earliestStart time.Time
		var alreadyOngoing bool
		if driverID != nil {
			earliestStart = driverEnd[*driverID]
			alreadyOngoing = driverOngoing[*driverID]
		} else {
			earliestStart = vehicleEnd[*vehicleID]
			alreadyOngoing = vehicleOngoing[*vehicleID]
		}

		// First third of blackouts are ongoing (span now), unless this resource already has one
		var start, end time.Time
		wantOngoing := i <= count/3 && !alreadyOngoing && earliestStart.Before(now) &&
			(driverID == nil || !activeDriverIDs[*driverID]) &&
			(vehicleID == nil || !activeVehicleIDs[*vehicleID])
		if wantOngoing {
			// Ongoing: started 0-3 days ago, ends 1-7 days from now
			start = now.AddDate(0, 0, -rand.Intn(3))
			end = now.AddDate(0, 0, 1+rand.Intn(7))
			if driverID != nil {
				driverOngoing[*driverID] = true
			} else {
				vehicleOngoing[*vehicleID] = true
			}
		} else {
			// Historical: placed in the past, after the previous blackout ended
			base := earliestStart
			if base.IsZero() {
				base = now.AddDate(0, 0, -(7 + rand.Intn(166)))
			}
			// Gap of 1-7 days after the previous blackout before the next one starts
			start = base.AddDate(0, 0, 1+rand.Intn(7))
			end = start.AddDate(0, 0, 1+rand.Intn(7))
			// If this pushes us into the future, anchor it further back instead
			if start.After(now) {
				start = now.AddDate(0, 0, -(7 + rand.Intn(166)))
				end = start.AddDate(0, 0, 1+rand.Intn(7))
			}
		}

		if driverID != nil {
			driverEnd[*driverID] = end
		} else {
			vehicleEnd[*vehicleID] = end
		}

		blackouts = append(blackouts, ResourceBlackout{
			ID:            i,
			DriverID:      driverID,
			VehicleID:     vehicleID,
			BlackoutStart: start,
			BlackoutEnd:   end,
			Reason:        reason,
		})
	}

	return blackouts
}

// GenerateInsertStatements generates a single INSERT statement for resource blackouts.
func GenerateInsertStatements(blackouts []ResourceBlackout) string {
	if len(blackouts) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(blackouts) * 150)
	sb.WriteString("INSERT INTO resource_blackouts (id, driver_id, vehicle_id, blackout_period, reason) VALUES\n")

	for i, b := range blackouts {
		driverVal := "NULL"
		if b.DriverID != nil {
			driverVal = fmt.Sprintf("%d", *b.DriverID)
		}
		vehicleVal := "NULL"
		if b.VehicleID != nil {
			vehicleVal = fmt.Sprintf("%d", *b.VehicleID)
		}

		sb.WriteString(fmt.Sprintf("    (%d, %s, %s, '[%s,%s)', '%s')",
			b.ID, driverVal, vehicleVal,
			b.BlackoutStart.Format("2006-01-02 15:04:05"),
			b.BlackoutEnd.Format("2006-01-02 15:04:05"),
			strings.ReplaceAll(b.Reason, "'", "''")))

		if i < len(blackouts)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}

	return sb.String()
}
