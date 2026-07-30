import { useCallback, useEffect, useMemo } from 'react';
import { useAtom, useAtomValue } from 'jotai';
import { Shipment, RoutePoint, Coordinates } from '@/model/shipments';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';
import { RouteContext } from './route-planner.types';
import {
  routePlannerContextAtom,
  routePlannerContextEntityAtom,
  routePlannerEntitySearchTermAtom,
  routePlannerSelectedShipmentAtom,
  routePlannerPlanningRouteAtom,
  routePlannerPendingPointTypeAtom,
} from './route-planner.store';
import {
  createRoutePoint,
  updateRoutePoints,
  generateOptimizedRoute,
  addRestStops,
} from './route-planner.utils';
import {
  convertDriverRouteToShipment,
  generateVehicleRouteShipments,
} from './route-planner.mocks';

export interface RoutePlannerActions {
  currentRoute: Shipment | null;
  isEditingAllowed: boolean;
  pendingPointType: RoutePoint['type'] | null;
  handleAddPointOfType: (type: RoutePoint['type']) => void;
  handleAddPoint: (coordinates: Coordinates, type: RoutePoint['type']) => void;
  handleRemovePoint: (pointId: string) => void;
  handleEditPoint: (point: RoutePoint) => void;
  handleReorderPoints: (points: RoutePoint[]) => void;
  handleOptimizeRoute: () => void;
  handleAddRestStops: () => void;
}

export const useRoutePlannerActions = (
  onShipmentUpdate?: (shipment: Shipment) => void,
): RoutePlannerActions => {
  const context = useAtomValue(routePlannerContextAtom);
  const [selectedShipment, setSelectedShipment] = useAtom(
    routePlannerSelectedShipmentAtom,
  );
  const [planningRoute, setPlanningRoute] = useAtom(
    routePlannerPlanningRouteAtom,
  );
  const [pendingPointType, setPendingPointType] = useAtom(
    routePlannerPendingPointTypeAtom,
  );

  const isEditingAllowed =
    context === 'active-shipments' || context === 'route-planning';

  const currentRoute =
    context === 'route-planning' ? planningRoute : selectedShipment;

  const updateCurrentRoute = useCallback(
    (updater: (route: Shipment) => Shipment) => {
      const updateFunction = (prev: Shipment | null) => {
        if (!prev) return prev;
        const updated = updater(prev);
        if (context !== 'route-planning') {
          onShipmentUpdate?.(updated);
        }
        return updated;
      };

      if (context === 'route-planning') {
        setPlanningRoute((prev) => updateFunction(prev) || prev);
      } else {
        setSelectedShipment((prev) => updateFunction(prev) || prev);
      }
    },
    [context, onShipmentUpdate, setPlanningRoute, setSelectedShipment],
  );

  const handleAddPointOfType = useCallback(
    (type: RoutePoint['type']) => {
      setPendingPointType(type);
    },
    [setPendingPointType],
  );

  const handleAddPoint = useCallback(
    (coordinates: Coordinates, type: RoutePoint['type']) => {
      const target = context === 'route-planning' ? planningRoute : selectedShipment;
      if (!target) return;

      const newPoint = createRoutePoint(coordinates, type, target);

      updateCurrentRoute((route) => {
        const newPoints = [...route.route.points, newPoint];
        return updateRoutePoints(route, newPoints);
      });

      setPendingPointType(null);
    },
    [context, planningRoute, selectedShipment, updateCurrentRoute, setPendingPointType],
  );

  const handleRemovePoint = useCallback(
    (pointId: string) => {
      updateCurrentRoute((route) => {
        const newPoints = route.route.points.filter((p) => p.id !== pointId);
        return updateRoutePoints(route, newPoints);
      });
    },
    [updateCurrentRoute],
  );

  const handleEditPoint = useCallback(
    (updatedPoint: RoutePoint) => {
      updateCurrentRoute((route) => {
        const newPoints = route.route.points.map((p) =>
          p.id === updatedPoint.id ? updatedPoint : p,
        );
        return updateRoutePoints(route, newPoints);
      });
    },
    [updateCurrentRoute],
  );

  const handleReorderPoints = useCallback(
    (newPoints: RoutePoint[]) => {
      updateCurrentRoute((route) => updateRoutePoints(route, newPoints));
    },
    [updateCurrentRoute],
  );

  const handleOptimizeRoute = useCallback(() => {
    updateCurrentRoute((route) => {
      const optimizedPoints = generateOptimizedRoute(route.route.points);
      return updateRoutePoints(route, optimizedPoints);
    });
  }, [updateCurrentRoute]);

  const handleAddRestStops = useCallback(() => {
    updateCurrentRoute((route) => {
      const pointsWithRest = addRestStops(route.route.points);
      return updateRoutePoints(route, pointsWithRest);
    });
  }, [updateCurrentRoute]);

  return {
    currentRoute,
    isEditingAllowed,
    pendingPointType,
    handleAddPointOfType,
    handleAddPoint,
    handleRemovePoint,
    handleEditPoint,
    handleReorderPoints,
    handleOptimizeRoute,
    handleAddRestStops,
  };
};

export interface UseRoutePlannerInitOptions {
  shipments: Shipment[];
  drivers: Driver[];
  vehicles: Vehicle[];
  initialContext?: RouteContext;
  initialContextEntity?: Driver | Vehicle;
}

export const useRoutePlannerInitialization = ({
  shipments,
  drivers,
  vehicles,
  initialContext,
  initialContextEntity,
}: UseRoutePlannerInitOptions) => {
  const [context, setContext] = useAtom(routePlannerContextAtom);
  const [contextEntity, setContextEntity] = useAtom(
    routePlannerContextEntityAtom,
  );
  const [entitySearchTerm, setEntitySearchTerm] = useAtom(
    routePlannerEntitySearchTermAtom,
  );
  const [selectedShipment, setSelectedShipment] = useAtom(
    routePlannerSelectedShipmentAtom,
  );
  const [planningRoute, setPlanningRoute] = useAtom(
    routePlannerPlanningRouteAtom,
  );

  useEffect(() => {
    if (initialContext) {
      setContext(initialContext);
    }
  }, [initialContext, setContext]);

  useEffect(() => {
    if (initialContextEntity) {
      setContextEntity(initialContextEntity);
      if ('name' in initialContextEntity) {
        setEntitySearchTerm(initialContextEntity.name);
      } else if ('plateNumber' in initialContextEntity) {
        setEntitySearchTerm(
          `${initialContextEntity.plateNumber} - ${initialContextEntity.make} ${initialContextEntity.model}`,
        );
      }
    }
  }, [initialContextEntity, setContextEntity, setEntitySearchTerm]);

  const contextualShipments = useMemo(() => {
    switch (context) {
      case 'route-planning':
        return [];
      case 'driver-routes':
        if (contextEntity && 'routes' in contextEntity) {
          return (contextEntity as Driver).routes.map((route) =>
            convertDriverRouteToShipment(route, contextEntity as Driver),
          );
        }
        return [];
      case 'vehicle-routes':
        if (contextEntity && 'plateNumber' in contextEntity) {
          return generateVehicleRouteShipments(contextEntity as Vehicle);
        }
        return [];
      case 'active-shipments':
      default:
        return shipments;
    }
  }, [context, contextEntity, shipments]);

  useEffect(() => {
    if (context === 'route-planning') {
      setSelectedShipment(planningRoute);
    } else if (contextualShipments.length > 0) {
      setSelectedShipment(contextualShipments[0]);
    } else {
      setSelectedShipment(null);
    }
  }, [context, contextualShipments, planningRoute, setSelectedShipment]);

  useEffect(() => {
    if (context === 'route-planning') return;

    const interval = setInterval(() => {
      setSelectedShipment((prev) => {
        if (!prev || prev.route.status !== 'active') return prev;

        const targetPoint = prev.route.points[0];
        if (!targetPoint) return prev;

        const currentLat = prev.route.vehicle.coordinates.lat;
        const currentLng = prev.route.vehicle.coordinates.lng;
        const targetLat = targetPoint.coordinates.lat;
        const targetLng = targetPoint.coordinates.lng;

        const newLat = currentLat + (targetLat - currentLat) * 0.01;
        const newLng = currentLng + (targetLng - currentLng) * 0.01;

        return {
          ...prev,
          route: {
            ...prev.route,
            vehicle: {
              ...prev.route.vehicle,
              coordinates: { lat: newLat, lng: newLng },
            },
          },
        };
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [context, setSelectedShipment]);

  return {
    contextualShipments,
    contextEntity,
  };
};
