import React, { useMemo } from 'react';
import { useAtom, useAtomValue } from 'jotai';
import { X, User, Truck } from 'lucide-react';
import { Driver } from '@/model/drivers';
import { Vehicle } from '@/model/vehicles';
import {
  routePlannerContextAtom,
  routePlannerContextEntityAtom,
  routePlannerEntitySearchTermAtom,
  routePlannerShowEntityDropdownAtom,
} from '../route-planner.store';
import { EntitySuggestion } from '../route-planner.types';
import { getEntityDisplayName } from '../route-planner.utils';

interface RouteEntitySelectorProps {
  drivers: Driver[];
  vehicles: Vehicle[];
  onContextChange?: (
    context: 'driver-routes' | 'vehicle-routes',
    entity?: Driver | Vehicle,
  ) => void;
}

export const RouteEntitySelector: React.FC<RouteEntitySelectorProps> = ({
  drivers,
  vehicles,
  onContextChange,
}) => {
  const context = useAtomValue(routePlannerContextAtom);
  const [contextEntity, setContextEntity] = useAtom(
    routePlannerContextEntityAtom,
  );
  const [entitySearchTerm, setEntitySearchTerm] = useAtom(
    routePlannerEntitySearchTermAtom,
  );
  const [showEntityDropdown, setShowEntityDropdown] = useAtom(
    routePlannerShowEntityDropdownAtom,
  );

  const suggestions: EntitySuggestion[] = useMemo(() => {
    if (
      context === 'active-shipments' ||
      context === 'route-planning' ||
      !entitySearchTerm
    ) {
      return [];
    }

    if (context === 'driver-routes') {
      return drivers
        .filter((driver) =>
          driver.name.toLowerCase().includes(entitySearchTerm.toLowerCase()),
        )
        .slice(0, 5)
        .map((driver) => ({
          id: driver.id,
          name: driver.name,
          type: 'driver' as const,
          entity: driver,
        }));
    }

    return vehicles
      .filter(
        (vehicle) =>
          vehicle.plateNumber
            .toLowerCase()
            .includes(entitySearchTerm.toLowerCase()) ||
          `${vehicle.make} ${vehicle.model}`
            .toLowerCase()
            .includes(entitySearchTerm.toLowerCase()),
      )
      .slice(0, 5)
      .map((vehicle) => ({
        id: vehicle.id,
        name: `${vehicle.plateNumber} - ${vehicle.make} ${vehicle.model}`,
        type: 'vehicle' as const,
        entity: vehicle,
      }));
  }, [context, entitySearchTerm, drivers, vehicles]);

  if (context !== 'driver-routes' && context !== 'vehicle-routes') {
    return null;
  }

  const handleSelect = (suggestion: EntitySuggestion) => {
    setContextEntity(suggestion.entity);
    setEntitySearchTerm(suggestion.name);
    setShowEntityDropdown(false);
    onContextChange?.(context, suggestion.entity);
  };

  const handleClear = () => {
    setContextEntity(undefined);
    setEntitySearchTerm('');
    onContextChange?.(context);
  };

  const label =
    context === 'driver-routes' ? 'Select Driver' : 'Select Vehicle';
  const placeholder =
    context === 'driver-routes' ? 'Search drivers...' : 'Search vehicles...';

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      <div className="relative">
        <input
          type="text"
          placeholder={placeholder}
          value={entitySearchTerm}
          onChange={(e) => {
            setEntitySearchTerm(e.target.value);
            setShowEntityDropdown(true);
          }}
          onFocus={() => setShowEntityDropdown(true)}
          className="w-full px-3 py-2 pr-8 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        />
        {contextEntity && (
          <button
            onClick={handleClear}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {showEntityDropdown && suggestions.length > 0 && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion.id}
              onClick={() => handleSelect(suggestion)}
              className="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none text-sm border-b border-gray-100 last:border-b-0"
            >
              <div className="flex items-center gap-2">
                {suggestion.type === 'driver' ? (
                  <User className="w-4 h-4 text-blue-500" />
                ) : (
                  <Truck className="w-4 h-4 text-purple-500" />
                )}
                <span className="font-medium">{suggestion.name}</span>
              </div>
              {suggestion.type === 'driver' && (
                <div className="text-xs text-gray-500 mt-1">
                  Status:{' '}
                  {(suggestion.entity as Driver).status.replace('-', ' ')}
                </div>
              )}
              {suggestion.type === 'vehicle' && (
                <div className="text-xs text-gray-500 mt-1">
                  {(suggestion.entity as Vehicle).year} •{' '}
                  {(suggestion.entity as Vehicle).status.replace('-', ' ')}
                </div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
