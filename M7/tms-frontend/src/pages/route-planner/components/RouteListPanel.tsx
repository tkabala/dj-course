import React, { useMemo } from 'react';
import { useAtom, useAtomValue } from 'jotai';
import { Route as RouteIcon, Truck, CheckCircle, AlertTriangle, Clock } from 'lucide-react';
import { Shipment } from '@/model/shipments';
import {
  routePlannerSearchTermAtom,
  routePlannerStatusFilterAtom,
  routePlannerSelectedShipmentAtom,
  routePlannerPendingPointTypeAtom,
} from '../route-planner.store';

interface RouteListPanelProps {
  shipments: Shipment[];
  title: string;
}

export const RouteListPanel: React.FC<RouteListPanelProps> = ({
  shipments,
  title,
}) => {
  const searchTerm = useAtomValue(routePlannerSearchTermAtom);
  const statusFilter = useAtomValue(routePlannerStatusFilterAtom);
  const [selectedShipment, setSelectedShipment] = useAtom(
    routePlannerSelectedShipmentAtom,
  );
  const [, setPendingPointType] = useAtom(routePlannerPendingPointTypeAtom);

  const filteredShipments = useMemo(() => {
    return shipments.filter((shipment) => {
      const matchesSearch =
        shipment.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        shipment.customer.toLowerCase().includes(searchTerm.toLowerCase()) ||
        shipment.route.vehicle.driver
          .toLowerCase()
          .includes(searchTerm.toLowerCase());

      const matchesStatus =
        statusFilter === 'all' || shipment.route.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [shipments, searchTerm, statusFilter]);

  const handleSelect = (shipment: Shipment) => {
    setSelectedShipment(shipment);
    setPendingPointType(null);
  };

  if (shipments.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mb-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <RouteIcon className="w-5 h-5 text-blue-600" />
        {title}
      </h2>

      <div className="max-h-80 overflow-y-auto pr-2 -mr-2">
        <div className="grid grid-cols-1 gap-3">
          {filteredShipments.map((shipment) => (
            <button
              key={shipment.id}
              onClick={() => handleSelect(shipment)}
              className={`p-3 rounded-lg border-2 transition-all text-left ${
                selectedShipment?.id === shipment.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-gray-900 text-sm">
                  {shipment.name}
                </h3>
                <div className="flex items-center">
                  {shipment.route.status === 'active' && (
                    <Truck className="w-4 h-4 text-green-600" />
                  )}
                  {shipment.route.status === 'completed' && (
                    <CheckCircle className="w-4 h-4 text-blue-600" />
                  )}
                  {shipment.route.status === 'delayed' && (
                    <AlertTriangle className="w-4 h-4 text-red-600" />
                  )}
                  {shipment.route.status === 'planned' && (
                    <Clock className="w-4 h-4 text-gray-600" />
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600 truncate">
                  {shipment.customer}
                </span>
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                    shipment.priority === 'urgent'
                      ? 'bg-red-100 text-red-800'
                      : shipment.priority === 'high'
                        ? 'bg-orange-100 text-orange-800'
                        : shipment.priority === 'medium'
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {shipment.priority.toUpperCase()}
                </span>
              </div>

              <div className="mt-2 text-xs text-gray-500">
                {shipment.route.points.length} stops •{' '}
                {shipment.route.totalDistance.toFixed(0)} km
              </div>
            </button>
          ))}
        </div>
      </div>

      {filteredShipments.length > 3 && (
        <div className="mt-2 text-xs text-gray-500 text-center">
          Scroll to see more ({filteredShipments.length} total)
        </div>
      )}
    </div>
  );
};
