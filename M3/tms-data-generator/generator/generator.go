package generator

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"tms-data-generator/generator/assignments"
	"tms-data-generator/generator/config"
	"tms-data-generator/generator/customers"
	"tms-data-generator/generator/driver_shifts"
	"tms-data-generator/generator/drivers"
	"tms-data-generator/generator/resource_blackouts"
	"tms-data-generator/generator/transportation_orders"
	"tms-data-generator/generator/vehicles"
)

func generateTimestampComment() string {
	// Go's reference time for formatting is "Mon Jan 2 15:04:05 MST 2006"
	return fmt.Sprintf("-- Generated on: %s\n", time.Now().Format("2006-01-02 15:04:05"))
}

// Generate generates the SQL file.
// It takes an output file path and generates SQL for the predefined data.
// It returns an error if writing the file fails.
func Generate(outputFile string) error {
	outputDir := filepath.Dir(outputFile)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("error creating output directory '%s': %w", outputDir, err)
	}

	schema, err := os.ReadFile("schema/create-tms-schema.sql")
	if err != nil {
		return fmt.Errorf("error reading schema file: %w", err)
	}

	var sb strings.Builder
	// start := time.Now() // Start timing

	// startVehicles := time.Now()
	// vehiclesStatements := vehicles.GenerateInsertStatements(vehicles.GenerateVehicles(config.VEHICLES))
	// fmt.Println("done generating vehicles", time.Now(), time.Since(startVehicles))

	// startDrivers := time.Now()
	// driversStatements := drivers.GenerateInsertStatements(drivers.GenerateDrivers(config.DRIVERS))
	// fmt.Println("done generating drivers", time.Now(), time.Since(startDrivers))

	var vehiclesStatements string
	var driversStatements string
	var customersStatements string
	wg := sync.WaitGroup{}

	start := time.Now() // Start timing

	// Phase 1: Generate independent entities in parallel
	wg.Add(3)
	go func() {
		defer wg.Done()
		startVehicles := time.Now()
		fmt.Println("Generating vehicles...", time.Now())
		vehiclesStatements = vehicles.GenerateInsertStatements(vehicles.GenerateVehicles(config.VEHICLES))
		fmt.Println("done generating vehicles", time.Now(), time.Since(startVehicles))
	}()
	go func() {
		defer wg.Done()
		startDrivers := time.Now()
		fmt.Println("Generating drivers...", time.Now())
		driversStatements = drivers.GenerateInsertStatements(drivers.GenerateDrivers(config.DRIVERS))
		fmt.Println("done generating drivers", time.Now(), time.Since(startDrivers))
	}()
	go func() {
		defer wg.Done()
		startCustomers := time.Now()
		fmt.Println("Generating customers...", time.Now())
		customersStatements = customers.GenerateInsertStatements(customers.GenerateCustomers(config.CUSTOMERS))
		fmt.Println("done generating customers", time.Now(), time.Since(startCustomers))
	}()

	fmt.Println("Waiting for independent entities...", time.Now())
	wg.Wait()

	// Phase 2: Generate orders (depends on customers)
	startOrders := time.Now()
	fmt.Println("Generating transportation orders...", time.Now())
	customersList := customers.GenerateCustomers(config.CUSTOMERS)
	ordersList := transportation_orders.GenerateTransportationOrders(config.TRANSPORTATION_ORDERS, customersList)
	fmt.Println("done generating transportation orders", time.Now(), time.Since(startOrders))

	// Phase 3: Generate order items
	startItems := time.Now()
	fmt.Println("Generating order items...", time.Now())
	orderItems := transportation_orders.GenerateOrderItems(ordersList)
	fmt.Println("done generating order items", time.Now(), time.Since(startItems))

	// Phase 4: Update order amounts based on items
	startUpdate := time.Now()
	fmt.Println("Updating order amounts...", time.Now())
	transportation_orders.UpdateOrderAmounts(ordersList, orderItems)
	fmt.Println("done updating order amounts", time.Now(), time.Since(startUpdate))

	// Phase 5: Generate timeline events
	startTimeline := time.Now()
	fmt.Println("Generating order timeline events...", time.Now())
	timelineEvents := transportation_orders.GenerateOrderTimelineEvents(ordersList)
	fmt.Println("done generating timeline events", time.Now(), time.Since(startTimeline))

	// Phase 6: Generate SQL statements
	startSQL := time.Now()
	fmt.Println("Generating SQL statements...", time.Now())
	ordersStatements := transportation_orders.GenerateInsertStatements(ordersList)
	timelineStatements := transportation_orders.GenerateTimelineEventsInsertStatements(timelineEvents)
	itemsStatements := transportation_orders.GenerateOrderItemsInsertStatements(orderItems)
	fmt.Println("done generating SQL statements", time.Now(), time.Since(startSQL))

	// Phase 7: Generate new domain entities in parallel
	driversList := drivers.GenerateDrivers(config.DRIVERS)
	vehiclesList := vehicles.GenerateVehicles(config.VEHICLES)

	// Shifts must be generated before assignments so we can filter on-shift drivers
	startShifts := time.Now()
	fmt.Println("Generating driver shifts...", time.Now())
	shiftsList := driver_shifts.GenerateDriverShifts(config.DRIVER_SHIFTS, driversList)
	driverShiftsStatements := driver_shifts.GenerateInsertStatements(shiftsList)
	fmt.Println("done generating driver shifts", time.Now(), time.Since(startShifts))

	startAssignments := time.Now()
	fmt.Println("Generating assignments...", time.Now())
	assignmentsList := assignments.GenerateAssignments(config.ASSIGNMENTS, ordersList, driversList, vehiclesList, shiftsList)
	assignmentsStatements := assignments.GenerateInsertStatements(assignmentsList)
	fmt.Println("done generating assignments", time.Now(), time.Since(startAssignments))

	// Collect drivers and vehicles with a currently active assignment so blackouts don't overlap them
	now := time.Now()
	activeDriverIDs := make(map[int]bool)
	activeVehicleIDs := make(map[int]bool)
	for _, a := range assignmentsList {
		if !a.BookingStart.After(now) && a.BookingEnd.After(now) {
			activeDriverIDs[a.DriverID] = true
			activeVehicleIDs[a.VehicleID] = true
		}
	}

	startBlackouts := time.Now()
	fmt.Println("Generating resource blackouts...", time.Now())
	blackoutsList := resource_blackouts.GenerateResourceBlackouts(config.RESOURCE_BLACKOUTS, driversList, vehiclesList, activeDriverIDs, activeVehicleIDs)
	blackoutsStatements := resource_blackouts.GenerateInsertStatements(blackoutsList)
	fmt.Println("done generating resource blackouts", time.Now(), time.Since(startBlackouts))

	elapsed := time.Since(start)
	fmt.Printf("Time taken to generate all: %s\n", elapsed)
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	sb.WriteString(generateTimestampComment())
	sb.WriteString(banner)
	sb.Write(schema)
	sb.WriteString("\n")
	sb.WriteString(vehiclesStatements)
	sb.WriteString(driversStatements)
	sb.WriteString(customersStatements)
	sb.WriteString(ordersStatements)
	sb.WriteString(timelineStatements)
	sb.WriteString(itemsStatements)
	sb.WriteString(driverShiftsStatements)
	sb.WriteString(assignmentsStatements)
	sb.WriteString(blackoutsStatements)

	err = os.WriteFile(outputFile, []byte(sb.String()), 0644)
	if err != nil {
		return fmt.Errorf("error writing file to '%s': %w", outputFile, err)
	}
	fmt.Println("SQL file generated:", outputFile)
	return nil
}
