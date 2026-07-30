import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { LoadingPage, ErrorMessage } from '@/components';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';
import { RoutePlannerHeader } from './components/RoutePlannerHeader';
import { RoutePlannerSidebar } from './components/RoutePlannerSidebar';
import { RouteMapSection } from './components/RouteMapSection';
import { useRoutePlannerData } from './route-planner.queries';
import { useRoutePlannerInitialization } from './route-planner.hooks';
import { RouteContext, RoutePlannerCallbacks } from './route-planner.types';

export const RoutePlannerPage: React.FC<RoutePlannerCallbacks> = ({
  onBack,
  onShipmentUpdate,
  onContextChange,
}) => {
  const [searchParams] = useSearchParams();

  const context =
    (searchParams.get('context') as RouteContext | null) || 'active-shipments';
  const entityId = searchParams.get('entityId');

  const { shipments, drivers, vehicles, isLoading, error, retry } =
    useRoutePlannerData();

  const initialContextEntity = entityId
    ? context === 'driver-routes'
      ? drivers.find((d) => d.id === entityId)
      : context === 'vehicle-routes'
        ? vehicles.find((v) => v.id === entityId)
        : undefined
    : undefined;

  useRoutePlannerInitialization({
    shipments,
    drivers,
    vehicles,
    initialContext: context,
    initialContextEntity,
  });

  if (isLoading) {
    return <LoadingPage />;
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <ErrorMessage
          error={
            error instanceof Error
              ? error.message
              : 'Failed to load route planner data'
          }
          onRetry={retry}
        />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <RoutePlannerHeader onBack={onBack} />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <RoutePlannerSidebar
          shipments={shipments}
          drivers={drivers}
          vehicles={vehicles}
          onShipmentUpdate={onShipmentUpdate}
          onContextChange={onContextChange}
        />
        <RouteMapSection onShipmentUpdate={onShipmentUpdate} />
      </div>
    </div>
  );
};
