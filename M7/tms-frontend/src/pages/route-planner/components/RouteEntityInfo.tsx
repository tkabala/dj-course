import React from 'react';
import { useAtomValue } from 'jotai';
import { User, Truck } from 'lucide-react';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';
import { routePlannerContextAtom, routePlannerContextEntityAtom } from '../route-planner.store';

export const RouteEntityInfo: React.FC = () => {
  const context = useAtomValue(routePlannerContextAtom);
  const contextEntity = useAtomValue(routePlannerContextEntityAtom);

  if (!contextEntity) {
    return null;
  }

  if (context === 'driver-routes') {
    const driver = contextEntity as Driver;
    return (
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <User className="w-4 h-4 text-blue-600" />
          Driver Information
        </h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Name:</span>
            <span className="font-medium">{driver.name}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Status:</span>
            <span className="font-medium capitalize">
              {driver.status.replace('-', ' ')}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Total Routes:</span>
            <span className="font-medium">{driver.routes.length}</span>
          </div>
        </div>
      </div>
    );
  }

  if (context === 'vehicle-routes') {
    const vehicle = contextEntity as Vehicle;
    return (
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Truck className="w-4 h-4 text-blue-600" />
          Vehicle Information
        </h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Plate:</span>
            <span className="font-medium">{vehicle.plateNumber}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Model:</span>
            <span className="font-medium">
              {vehicle.make} {vehicle.model}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Status:</span>
            <span className="font-medium capitalize">
              {vehicle.status.replace('-', ' ')}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Mileage:</span>
            <span className="font-medium">
              {vehicle.mileage.toLocaleString()} km
            </span>
          </div>
        </div>
      </div>
    );
  }

  return null;
};
