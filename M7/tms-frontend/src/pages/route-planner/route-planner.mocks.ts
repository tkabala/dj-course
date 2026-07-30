import { RoutePoint, Vehicle, RouteData, Shipment } from '@/model/shipments';
import { Driver, DriverRoute } from '@/model/drivers';
import { Vehicle as VehicleType } from '@/model/vehicles';

export const convertDriverRouteToShipment = (
  driverRoute: DriverRoute,
  driver: Driver,
): Shipment => {
  const routePoints: RoutePoint[] = driverRoute.points.map((point, index) => ({
    id: `${driverRoute.id}-point-${index}`,
    coordinates: { lat: point.lat, lng: point.lng },
    type:
      point.type === 'start'
        ? 'pickup'
        : point.type === 'end'
          ? 'delivery'
          : 'rest',
    name: point.name,
    address: point.name,
    estimatedArrival: point.timestamp,
    duration: 60,
    notes: `Driver route point - ${point.type}`,
  }));

  const vehicle: Vehicle = {
    id: `driver-vehicle-${driver.id}`,
    coordinates: driver.currentLocation
      ? {
          lat: driver.currentLocation.lat,
          lng: driver.currentLocation.lng,
        }
      : { lat: 52.2297, lng: 21.0122 },
    heading: 180,
    speed: driverRoute.status === 'active' ? 75 : 0,
    driver: driver.name,
    plateNumber: 'DRIVER-VEHICLE',
  };

  const routeData: RouteData = {
    id: driverRoute.id,
    name: driverRoute.name,
    points: routePoints,
    vehicle,
    totalDistance: driverRoute.distance,
    estimatedDuration: Math.floor((driverRoute.distance / 80) * 60),
    status:
      driverRoute.status === 'active'
        ? 'active'
        : driverRoute.status === 'completed'
          ? 'completed'
          : 'planned',
    startTime: driverRoute.startDate,
    estimatedCompletion: driverRoute.endDate,
  };

  return {
    id: `shipment-${driverRoute.id}`,
    name: driverRoute.name,
    customer: `Driver Route - ${driver.name}`,
    priority: driverRoute.status === 'active' ? 'high' : 'medium',
    route: routeData,
    createdAt: driverRoute.startDate,
    dueDate: driverRoute.endDate,
  };
};

export const generateVehicleRouteShipments = (
  vehicle: VehicleType,
): Shipment[] => {
  const routes: Shipment[] = [];
  const now = new Date();

  for (let i = 0; i < 8; i++) {
    const startDate = new Date(
      now.getTime() - (i * 5 + Math.random() * 3) * 24 * 60 * 60 * 1000,
    );
    const endDate = new Date(
      startDate.getTime() + (1 + Math.random() * 2) * 24 * 60 * 60 * 1000,
    );

    const origins = ['Warszawa', 'Kraków', 'Gdańsk', 'Wrocław', 'Poznań'];
    const destinations = ['Łódź', 'Szczecin', 'Lublin', 'Katowice', 'Bydgoszcz'];

    const origin = origins[Math.floor(Math.random() * origins.length)];
    const destination =
      destinations[Math.floor(Math.random() * destinations.length)];
    const distance = Math.floor(Math.random() * 800 + 200);

    const routePoints: RoutePoint[] = [
      {
        id: `vehicle-route-${vehicle.id}-${i}-start`,
        coordinates: {
          lat: 52.2297 + (Math.random() - 0.5) * 2,
          lng: 21.0122 + (Math.random() - 0.5) * 4,
        },
        type: 'pickup',
        name: origin,
        address: `${origin}, Polska`,
        estimatedArrival: startDate,
        duration: 60,
      },
      {
        id: `vehicle-route-${vehicle.id}-${i}-end`,
        coordinates: {
          lat: 50.0647 + (Math.random() - 0.5) * 2,
          lng: 19.945 + (Math.random() - 0.5) * 4,
        },
        type: 'delivery',
        name: destination,
        address: `${destination}, Polska`,
        estimatedArrival: endDate,
        duration: 45,
      },
    ];

    const vehicleForRoute: Vehicle = {
      id: vehicle.id,
      coordinates: vehicle.currentLocation
        ? {
            lat: vehicle.currentLocation.lat,
            lng: vehicle.currentLocation.lng,
          }
        : { lat: 52.2297, lng: 21.0122 },
      heading: 180,
      speed: i === 0 ? 85 : 0,
      driver: vehicle.currentDriver || 'Unassigned',
      plateNumber: vehicle.plateNumber,
    };

    const routeData: RouteData = {
      id: `vehicle-route-${vehicle.id}-${i}`,
      name: `${origin} → ${destination}`,
      points: routePoints,
      vehicle: vehicleForRoute,
      totalDistance: distance,
      estimatedDuration: Math.floor((distance / 80) * 60),
      status: i === 0 ? 'active' : i < 3 ? 'completed' : 'completed',
      startTime: startDate,
      estimatedCompletion: endDate,
    };

    routes.push({
      id: `vehicle-shipment-${vehicle.id}-${i}`,
      name: `${origin} → ${destination}`,
      customer: `Vehicle Route - ${vehicle.plateNumber}`,
      priority: i === 0 ? 'high' : 'medium',
      route: routeData,
      createdAt: startDate,
      dueDate: endDate,
    });
  }

  return routes.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
};
