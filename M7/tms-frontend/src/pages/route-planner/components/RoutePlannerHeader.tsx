import React from 'react';
import { ArrowLeft } from 'lucide-react';
import { useAtomValue } from 'jotai';
import { routePlannerContextAtom, routePlannerContextEntityAtom } from '../route-planner.store';
import { getContextTitle, getContextDescription, getContextIcon } from '../route-planner.utils';

interface RoutePlannerHeaderProps {
  onBack?: () => void;
}

export const RoutePlannerHeader: React.FC<RoutePlannerHeaderProps> = ({
  onBack,
}) => {
  const context = useAtomValue(routePlannerContextAtom);
  const contextEntity = useAtomValue(routePlannerContextEntityAtom);

  return (
    <div className="mb-6">
      {onBack && (
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-blue-600 hover:text-blue-800 mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
      )}
      <div className="flex items-center gap-3">
        {getContextIcon(context)}
        <div>
          <h2 className="text-2xl font-bold text-gray-900">
            {getContextTitle(context, contextEntity)}
          </h2>
          <p className="text-gray-600">
            {getContextDescription(context, contextEntity)}
          </p>
        </div>
      </div>
    </div>
  );
};
