package driver_shifts

import (
	"fmt"
	"math/rand"
	"strings"

	"tms-data-generator/generator/drivers"
)

type shiftTemplate struct {
	Start string
	End   string
}

var shiftTemplates = []shiftTemplate{
	{"06:00:00", "14:00:00"},
	{"08:00:00", "16:00:00"},
	{"10:00:00", "18:00:00"},
	{"14:00:00", "22:00:00"},
	{"22:00:00", "06:00:00"},
}

// GenerateDriverShifts generates recurring shift schedules for drivers.
// Each (driver, day-of-week) pair gets at most one shift to prevent overlaps.
func GenerateDriverShifts(count int, driversList []drivers.Driver) []DriverShift {
	type driverDay struct{ driverID, dayOfWeek int }
	seen := make(map[driverDay]bool)

	shifts := make([]DriverShift, 0, count)
	id := 1
	for i := 0; i < count; i++ {
		// Find a (driver, day) combo not yet used
		var driver drivers.Driver
		var dow int
		found := false
		for attempt := 0; attempt < 20; attempt++ {
			driver = driversList[rand.Intn(len(driversList))]
			dow = rand.Intn(7)
			if !seen[driverDay{driver.ID, dow}] {
				found = true
				break
			}
		}
		if !found {
			continue
		}
		seen[driverDay{driver.ID, dow}] = true

		template := shiftTemplates[rand.Intn(len(shiftTemplates))]
		shifts = append(shifts, DriverShift{
			ID:        id,
			DriverID:  driver.ID,
			DayOfWeek: dow,
			StartTime: template.Start,
			EndTime:   template.End,
		})
		id++
	}
	return shifts
}

// GenerateInsertStatements generates a single INSERT statement for driver shifts.
func GenerateInsertStatements(shifts []DriverShift) string {
	if len(shifts) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(shifts) * 80)
	sb.WriteString("INSERT INTO driver_shifts (id, driver_id, day_of_week, start_time, end_time) VALUES\n")

	for i, s := range shifts {
		sb.WriteString(fmt.Sprintf("    (%d, %d, %d, '%s', '%s')",
			s.ID, s.DriverID, s.DayOfWeek, s.StartTime, s.EndTime))

		if i < len(shifts)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}

	return sb.String()
}
