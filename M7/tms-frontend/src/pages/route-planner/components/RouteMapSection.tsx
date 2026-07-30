import React from 'react';
import { MapPin } from 'lucide-react';
import { useAtomValue } from 'jotai';
import { LogisticsMap } from '../LogisticsMap';
import { RouteSummary } from '../RouteSummary';
import { useRoutePlannerActions } from '../route-planner.hooks';
import { routePlannerContextAtom } from '../route-planner.store';

interface RouteMapSectionProps {
  onShipmentUpdate?: (shipment: import('@/model/shipments').Shipment) => void;
}

export const RouteMapSection: React.FC<RouteMapSectionProps> = ({
  onShipmentUpdate,
}) => {
  const context = useAtomValue(routePlannerContextAtom);
  const {
    currentRoute,
    isEditingAllowed,
    pendingPointType,
    handleAddPoint,
    handleRemovePoint,
    handleEditPoint,
    handleReorderPoints,
  } = useRoutePlannerActions(onShipmentUpdate);

  const getEmptyMessage = () => {
    switch (context) {
      case 'active-shipments':
        return 'Select a shipment to view its route';
      case 'driver-routes':
        return 'Select a driver and route to view on the map';
      case 'vehicle-routes':
        return 'Select a vehicle and route to view on the map';
      default:
        return 'Start planning your route by adding points to the map';
    }
  };

  return (
    <div className="lg:col-span-3 space-y-6">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="h-[600px]">
          {currentRoute ? (
            <LogisticsMap
              points={currentRoute.route.points}
              vehicle={currentRoute.route.vehicle}
              onPointAdd={isEditingAllowed ? handleAddPoint : undefined}
              onPointRemove={isEditingAllowed ? handleRemovePoint : undefined}
              onPointEdit={isEditingAllowed ? handleEditPoint : undefined}
              pendingPointType={isEditingAllowed ? pendingPointType : null}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-50">
              <div className="text-center">
                <MapPin className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  No Route Selected
                </h3>
                <p className="text-gray-500">{getEmptyMessage()}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {currentRoute && (
        <RouteSummary
          route={currentRoute.route}
          onReorderPoints={isEditingAllowed ? handleReorderPoints : undefined}
          allowReordering={isEditingAllowed}
        />
      )}
    </div>
  );
};
