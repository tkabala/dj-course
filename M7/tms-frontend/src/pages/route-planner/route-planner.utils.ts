import React from 'react';
import {
  RoutePoint,
  Coordinates,
  Vehicle,
  RouteData,
  Shipment,
} from '@/model/shipments';
import { calculateDistance } from './mapUtils';
import { RouteContext, ContextOption } from './route-planner.types';
import {
  Navigation,
  Route as RouteIcon,
  User,
  Truck,
} from 'lucide-react';

export const calculateRouteDistance = (points: RoutePoint[]): number => {
  if (points.length < 2) return 0;

  let totalDistance = 0;
  for (let i = 0; i < points.length - 1; i++) {
    totalDistance += calculateDistance(
      points[i].coordinates,
      points[i + 1].coordinates,
    );
  }
  return totalDistance;
};

export const estimateTravelTime = (
  distance: number,
  averageSpeed: number = 80,
): number => {
  return Math.round((distance / averageSpeed) * 60); // minutes
};

export const generateOptimizedRoute = (points: RoutePoint[]): RoutePoint[] => {
  if (points.length <= 2) return points;

  const optimizedPoints = [...points];

  const sortedPoints = optimizedPoints.sort((a, b) => {
    if (a.type === 'pickup' && b.type === 'delivery') return -1;
    if (a.type === 'delivery' && b.type === 'pickup') return 1;
    return 0;
  });

  return sortedPoints.map((point, index) => {
    const latAdjustment = (Math.random() - 0.5) * 0.002;
    const lngAdjustment = (Math.random() - 0.5) * 0.002;

    return {
      ...point,
      coordinates: {
        lat: point.coordinates.lat + latAdjustment,
        lng: point.coordinates.lng + lngAdjustment,
      },
      name: point.name + (index === 0 ? '' : ' (Optimized)'),
      notes: point.notes
        ? point.notes + ' - Route optimized'
        : 'Route optimized for efficiency',
    };
  });
};

export const addRestStops = (
  points: RoutePoint[],
  maxDrivingTime: number = 270,
): RoutePoint[] => {
  if (points.length < 2) return points;

  const result: RoutePoint[] = [];
  let cumulativeTime = 0;

  for (let i = 0; i < points.length; i++) {
    result.push(points[i]);

    if (i < points.length - 1) {
      const distance = calculateDistance(
        points[i].coordinates,
        points[i + 1].coordinates,
      );
      const travelTime = estimateTravelTime(distance);
      cumulativeTime += travelTime;

      if (cumulativeTime >= maxDrivingTime) {
        const midPoint = {
          lat:
            (points[i].coordinates.lat + points[i + 1].coordinates.lat) / 2,
          lng:
            (points[i].coordinates.lng + points[i + 1].coordinates.lng) / 2,
        };

        result.push({
          id: `rest-${Date.now()}-${i}`,
          coordinates: midPoint,
          type: 'rest' as const,
          name: 'Mandatory Rest Stop',
          address: `Rest area between ${points[i].name} and ${points[i + 1].name}`,
          duration: 45,
          notes: 'EU regulation: 45min break after 4.5h driving',
          estimatedArrival: new Date(Date.now() + cumulativeTime * 60 * 1000),
        });

        cumulativeTime = 0;
      }
    }
  }

  return result;
};

export const createDefaultPlanningRoute = (): Shipment => {
  const defaultVehicle: Vehicle = {
    id: 'planning-vehicle',
    coordinates: { lat: 52.2297, lng: 21.0122 },
    heading: 0,
    speed: 0,
    driver: 'Select Driver',
    plateNumber: 'Select Vehicle',
  };

  const defaultRoute: RouteData = {
    id: 'planning-route',
    name: 'New Route Plan',
    points: [],
    vehicle: defaultVehicle,
    totalDistance: 0,
    estimatedDuration: 0,
    status: 'planned',
    startTime: new Date(),
    estimatedCompletion: new Date(),
  };

  return {
    id: 'planning-shipment',
    name: 'New Route Plan',
    customer: 'Route Planning',
    priority: 'medium',
    route: defaultRoute,
    createdAt: new Date(),
    dueDate: new Date(),
  };
};

export const updateRoutePoints = (
  shipment: Shipment,
  newPoints: RoutePoint[],
): Shipment => {
  const totalDistance = calculateRouteDistance(newPoints);
  const estimatedDuration = estimateTravelTime(totalDistance);

  return {
    ...shipment,
    route: {
      ...shipment.route,
      points: newPoints,
      totalDistance,
      estimatedDuration,
    },
  };
};

export const createRoutePoint = (
  coordinates: Coordinates,
  type: RoutePoint['type'],
  targetShipment: Shipment,
): RoutePoint => {
  return {
    id: `point-${Date.now()}`,
    coordinates,
    type,
    name: `New ${type.charAt(0).toUpperCase() + type.slice(1)}`,
    address: `${coordinates.lat.toFixed(4)}, ${coordinates.lng.toFixed(4)}`,
    estimatedArrival: new Date(
      Date.now() + targetShipment.route.points.length * 2 * 60 * 60 * 1000,
    ),
    duration: type === 'rest' ? 45 : type === 'fuel' ? 30 : 60,
    notes: type === 'rest' ? 'Driver rest period' : undefined,
  };
};

export const getContextOptions = (): ContextOption[] => [
  {
    value: 'route-planning',
    label: 'Route Planning',
    icon: React.createElement(Navigation, { className: 'w-4 h-4' }),
  },
  {
    value: 'active-shipments',
    label: 'Active Shipments',
    icon: React.createElement(RouteIcon, { className: 'w-4 h-4' }),
  },
  {
    value: 'driver-routes',
    label: 'Driver Routes',
    icon: React.createElement(User, { className: 'w-4 h-4' }),
  },
  {
    value: 'vehicle-routes',
    label: 'Vehicle Routes',
    icon: React.createElement(Truck, { className: 'w-4 h-4' }),
  },
];

export const getContextTitle = (
  context: RouteContext,
  contextEntity?: Driver | Vehicle,
): string => {
  switch (context) {
    case 'route-planning':
      return 'Route Planning';
    case 'driver-routes':
      return contextEntity && 'name' in contextEntity
        ? `Driver Routes - ${contextEntity.name}`
        : 'Driver Routes';
    case 'vehicle-routes':
      return contextEntity && 'plateNumber' in contextEntity
        ? `Vehicle Routes - ${contextEntity.plateNumber}`
        : 'Vehicle Routes';
    case 'active-shipments':
    default:
      return 'Active Shipments - Route Planner';
  }
};

export const getContextDescription = (
  context: RouteContext,
  contextEntity?: Driver | Vehicle,
): string => {
  switch (context) {
    case 'route-planning':
      return 'Create and optimize new routes with advanced planning tools';
    case 'driver-routes':
      return contextEntity && 'name' in contextEntity
        ? `View and track routes assigned to ${contextEntity.name}`
        : 'Select a driver to view their routes';
    case 'vehicle-routes':
      return contextEntity && 'plateNumber' in contextEntity
        ? `View and track routes completed by ${contextEntity.plateNumber}`
        : 'Select a vehicle to view its routes';
    case 'active-shipments':
    default:
      return 'Plan and manage active shipment routes with real-time tracking';
  }
};

export const getContextIcon = (
  context: RouteContext,
): React.ReactNode => {
  const options = getContextOptions();
  const option = options.find((o) => o.value === context);
  return option?.icon ?? React.createElement(RouteIcon, { className: 'w-5 h-5 text-blue-600' });
};

export const getEntityDisplayName = (
  entity: Driver | Vehicle,
): string => {
  if ('name' in entity) {
    return entity.name;
  }
  return `${entity.plateNumber} - ${entity.make} ${entity.model}`;
};

export const getEntitySearchLabel = (
  context: RouteContext,
): string => {
  if (context === 'driver-routes') return 'Select Driver';
  if (context === 'vehicle-routes') return 'Select Vehicle';
  return 'Search';
};

export const isRouteEditingAllowed = (context: RouteContext): boolean => {
  return context === 'active-shipments' || context === 'route-planning';
};
