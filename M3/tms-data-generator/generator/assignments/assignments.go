package assignments

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"

	"tms-data-generator/generator/driver_shifts"
	"tms-data-generator/generator/drivers"
	"tms-data-generator/generator/transportation_orders"
	"tms-data-generator/generator/vehicles"
)

// onShiftAt returns true if the driver has a shift covering the given time.
func onShiftAt(driverID int, at time.Time, shiftsList []driver_shifts.DriverShift) bool {
	dow := int(at.Weekday())
	nowStr := at.Format("15:04:05")
	for _, s := range shiftsList {
		if s.DriverID != driverID {
			continue
		}
		if s.StartTime <= s.EndTime {
			// Regular shift (e.g. 08:00–16:00)
			if s.DayOfWeek == dow && s.StartTime <= nowStr && nowStr < s.EndTime {
				return true
			}
		} else {
			// Overnight shift (e.g. 22:00–06:00)
			prevDOW := (dow + 6) % 7
			if (s.DayOfWeek == dow && nowStr >= s.StartTime) ||
				(s.DayOfWeek == prevDOW && nowStr < s.EndTime) {
				return true
			}
		}
	}
	return false
}

// GenerateAssignments generates trip assignments with non-overlapping booking periods per driver and vehicle.
// Active orders (IN_TRANSIT, PROCESSING, etc.) always get ongoing assignments that span now.
// Historical orders fill the remaining slots with past booking periods.
func GenerateAssignments(count int, ordersList []transportation_orders.TransportationOrder, driversList []drivers.Driver, vehiclesList []vehicles.Vehicle, shiftsList []driver_shifts.DriverShift) []Assignment {
	now := time.Now()

	activeStatuses := map[transportation_orders.OrderStatus]bool{
		transportation_orders.OrderInTransit:      true,
		transportation_orders.OrderProcessing:     true,
		transportation_orders.OrderPending:        true,
		transportation_orders.OrderReadyForPickup: true,
	}

	var activeOrders, historicalOrders []transportation_orders.TransportationOrder
	for _, o := range ordersList {
		if activeStatuses[o.Status] {
			activeOrders = append(activeOrders, o)
		} else {
			historicalOrders = append(historicalOrders, o)
		}
	}

	// Sort historical orders by date for chronological chaining
	sort.Slice(historicalOrders, func(i, j int) bool {
		return historicalOrders[i].OrderDate.Before(historicalOrders[j].OrderDate)
	})

	assignments := make([]Assignment, 0, count)
	driverEnd := make(map[int]time.Time)
	vehicleEnd := make(map[int]time.Time)
	id := 1

	// Build list of drivers currently on shift
	var onShiftDrivers []drivers.Driver
	for _, d := range driversList {
		if onShiftAt(d.ID, now, shiftsList) {
			onShiftDrivers = append(onShiftDrivers, d)
		}
	}

	// Step 1: assign active orders first — each gets a unique on-shift driver+vehicle with a booking spanning now
	maxCurrent := len(activeOrders)
	if maxCurrent > len(onShiftDrivers) {
		maxCurrent = len(onShiftDrivers)
	}
	if maxCurrent > len(vehiclesList) {
		maxCurrent = len(vehiclesList)
	}
	if maxCurrent > count {
		maxCurrent = count
	}

	for i := 0; i < maxCurrent; i++ {
		order := activeOrders[i]
		driver := onShiftDrivers[i]
		vehicle := vehiclesList[i]

		start := now.Add(-time.Duration(1+rand.Intn(4)) * time.Hour)
		end := now.Add(time.Duration(4+rand.Intn(21)) * time.Hour) // 4–24h into the future

		assignments = append(assignments, Assignment{
			ID:           id,
			OrderID:      order.ID,
			DriverID:     driver.ID,
			VehicleID:    vehicle.ID,
			BookingStart: start,
			BookingEnd:   end,
		})
		driverEnd[driver.ID] = end
		vehicleEnd[vehicle.ID] = end
		id++
	}

	// Step 2: fill remaining slots with historical assignments
	for i, order := range historicalOrders {
		if id > count {
			break
		}

		driver := driversList[i%len(driversList)]
		vehicle := vehiclesList[i%len(vehiclesList)]

		start := order.OrderDate
		if de, ok := driverEnd[driver.ID]; ok && de.After(start) {
			start = de
		}
		if ve, ok := vehicleEnd[vehicle.ID]; ok && ve.After(start) {
			start = ve
		}

		// Skip if chaining pushed start into the future (avoid mixing future historical assignments)
		if start.After(now) {
			continue
		}

		duration := time.Duration(2+rand.Intn(7)) * time.Hour
		end := start.Add(duration)

		assignments = append(assignments, Assignment{
			ID:           id,
			OrderID:      order.ID,
			DriverID:     driver.ID,
			VehicleID:    vehicle.ID,
			BookingStart: start,
			BookingEnd:   end,
		})
		driverEnd[driver.ID] = end
		vehicleEnd[vehicle.ID] = end
		id++
	}

	return assignments
}

// GenerateInsertStatements generates a single INSERT statement for assignments.
func GenerateInsertStatements(assignments []Assignment) string {
	if len(assignments) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(assignments) * 150)
	sb.WriteString("INSERT INTO assignments (id, order_id, driver_id, vehicle_id, booking_period) VALUES\n")

	for i, a := range assignments {
		sb.WriteString(fmt.Sprintf("    (%d, %d, %d, %d, '[%s,%s)')",
			a.ID, a.OrderID, a.DriverID, a.VehicleID,
			a.BookingStart.Format("2006-01-02 15:04:05"),
			a.BookingEnd.Format("2006-01-02 15:04:05")))

		if i < len(assignments)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}

	return sb.String()
}
