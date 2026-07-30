import { useShipmentsList, useDriversList, useVehiclesList } from '@/hooks/queries';
import { RoutePlannerData } from './route-planner.types';

export const useRoutePlannerData = (): RoutePlannerData => {
  const {
    data: shipments = [],
    isLoading: shipmentsLoading,
    error: shipmentsError,
    refetch: refetchShipments,
  } = useShipmentsList();

  const {
    data: drivers = [],
    isLoading: driversLoading,
    error: driversError,
    refetch: refetchDrivers,
  } = useDriversList();

  const {
    data: vehicles = [],
    isLoading: vehiclesLoading,
    error: vehiclesError,
    refetch: refetchVehicles,
  } = useVehiclesList();

  const isLoading = shipmentsLoading || driversLoading || vehiclesLoading;
  const error = shipmentsError || driversError || vehiclesError;

  const retry = () => {
    refetchShipments();
    refetchDrivers();
    refetchVehicles();
  };

  return {
    shipments,
    drivers,
    vehicles,
    isLoading,
    error,
    retry,
  };
};
