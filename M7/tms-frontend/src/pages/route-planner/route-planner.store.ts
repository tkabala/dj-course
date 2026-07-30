import { atom } from 'jotai';
import { RouteContext, RouteStatusFilter } from './route-planner.types';
import { Shipment } from '@/model/shipments';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';
import { createDefaultPlanningRoute } from './route-planner.utils';

export const routePlannerContextAtom = atom<RouteContext>('active-shipments');

export const routePlannerContextEntityAtom = atom<Driver | Vehicle | undefined>(
  undefined,
);

export const routePlannerSearchTermAtom = atom<string>('');

export const routePlannerStatusFilterAtom = atom<RouteStatusFilter>('all');

export const routePlannerEntitySearchTermAtom = atom<string>('');

export const routePlannerShowEntityDropdownAtom = atom<boolean>(false);

export const routePlannerSelectedShipmentAtom = atom<Shipment | null>(null);

export const routePlannerPendingPointTypeAtom = atom<
  Shipment['route']['points'][number]['type'] | null
>(null);

export const routePlannerPlanningRouteAtom = atom<Shipment>(
  createDefaultPlanningRoute(),
);
