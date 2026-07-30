import React from 'react';
import { Shipment } from '@/model/shipments';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';

export type RouteContext =
  | 'active-shipments'
  | 'driver-routes'
  | 'vehicle-routes'
  | 'route-planning';

export type RouteStatusFilter =
  | 'all'
  | 'active'
  | 'completed'
  | 'planned'
  | 'delayed';

export interface ContextOption {
  value: RouteContext;
  label: string;
  icon: React.ReactNode;
}

export interface EntitySuggestion {
  id: string;
  name: string;
  type: 'driver' | 'vehicle';
  entity: Driver | Vehicle;
}

export interface RoutePlannerData {
  shipments: Shipment[];
  drivers: Driver[];
  vehicles: Vehicle[];
  isLoading: boolean;
  error: Error | null;
  retry: () => void;
}

export interface RoutePlannerRoute {
  route: Shipment;
}

export interface RoutePlannerCallbacks {
  onBack?: () => void;
  onShipmentUpdate?: (shipment: Shipment) => void;
  onContextChange?: (context: RouteContext, entity?: Driver | Vehicle) => void;
}
