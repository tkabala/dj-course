import React, { useMemo } from 'react';
import { useAtomValue } from 'jotai';
import { Route as RouteIcon, User, Truck } from 'lucide-react';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';
import { Shipment } from '@/model/shipments';
import { RouteContextSelector } from './RouteContextSelector';
import { RouteEntitySelector } from './RouteEntitySelector';
import { RouteFilters } from './RouteFilters';
import { RouteControls } from '../RouteControls';
import { VehicleStatus } from '../VehicleStatus';
import { RouteListPanel } from './RouteListPanel';
import { RouteEntityInfo } from './RouteEntityInfo';
import { useRoutePlannerActions } from '../route-planner.hooks';
import {
  routePlannerContextAtom,
  routePlannerContextEntityAtom,
  routePlannerSelectedShipmentAtom,
} from '../route-planner.store';
import {
  convertDriverRouteToShipment,
  generateVehicleRouteShipments,
} from '../route-planner.mocks';
import { isRouteEditingAllowed } from '../route-planner.utils';

interface RoutePlannerSidebarProps {
  shipments: Shipment[];
  drivers: Driver[];
  vehicles: Vehicle[];
  onShipmentUpdate?: (shipment: Shipment) => void;
  onContextChange?: (
    context: import('../route-planner.types').RouteContext,
    entity?: Driver | Vehicle,
  ) => void;
}

export const RoutePlannerSidebar: React.FC<RoutePlannerSidebarProps> = ({
  shipments,
  drivers,
  vehicles,
  onShipmentUpdate,
  onContextChange,
}) => {
  const context = useAtomValue(routePlannerContextAtom);
  const contextEntity = useAtomValue(routePlannerContextEntityAtom);
  const selectedShipment = useAtomValue(routePlannerSelectedShipmentAtom);
  const { currentRoute, handleAddPointOfType, handleOptimizeRoute, handleAddRestStops } =
    useRoutePlannerActions(onShipmentUpdate);

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

  const listTitle =
    context === 'active-shipments'
      ? 'Active Shipments'
      : context === 'driver-routes'
        ? 'Driver Routes'
        : 'Vehicle Routes';

  const hasValidData =
    (context === 'route-planning' && currentRoute) ||
    (contextualShipments.length > 0 && selectedShipment);

  return (
    <div className="lg:col-span-1 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-4">
        <div className="space-y-4">
          <RouteContextSelector />
          <RouteEntitySelector
            drivers={drivers}
            vehicles={vehicles}
            onContextChange={onContextChange}
          />
          {context !== 'route-planning' && <RouteFilters />}
        </div>
      </div>

      {currentRoute && isRouteEditingAllowed(context) && (
        <RouteControls
          route={currentRoute.route}
          onAddPoint={handleAddPointOfType}
          onOptimizeRoute={handleOptimizeRoute}
          onAddRestStops={handleAddRestStops}
        />
      )}

      {currentRoute && <VehicleStatus vehicle={currentRoute.route.vehicle} />}

      {context !== 'route-planning' && hasValidData && (
        <RouteListPanel shipments={contextualShipments} title={listTitle} />
      )}

      <RouteEntityInfo />

      {!hasValidData && context !== 'route-planning' && (
        <div className="bg-white rounded-lg shadow-lg p-6 text-center">
          <div className="text-gray-400 mb-2">
            {context === 'driver-routes' ? (
              <User className="w-8 h-8 mx-auto" />
            ) : context === 'vehicle-routes' ? (
              <Truck className="w-8 h-8 mx-auto" />
            ) : (
              <RouteIcon className="w-8 h-8 mx-auto" />
            )}
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {context === 'driver-routes' && !contextEntity
              ? 'Select a Driver'
              : context === 'vehicle-routes' && !contextEntity
                ? 'Select a Vehicle'
                : 'No Routes Available'}
          </h3>
          <p className="text-gray-500 text-sm">
            {context === 'driver-routes' && !contextEntity
              ? 'Choose a driver to view their routes'
              : context === 'vehicle-routes' && !contextEntity
                ? 'Choose a vehicle to view its routes'
                : 'No routes found for the selected criteria'}
          </p>
        </div>
      )}
    </div>
  );
};
