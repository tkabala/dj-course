import React from 'react';
import { useAtom } from 'jotai';
import { routePlannerContextAtom } from '../route-planner.store';
import { getContextOptions } from '../route-planner.utils';

export const RouteContextSelector: React.FC = () => {
  const [context, setContext] = useAtom(routePlannerContextAtom);
  const contextOptions = getContextOptions();

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Context
      </label>
      <select
        value={context}
        onChange={(e) => setContext(e.target.value as typeof context)}
        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
      >
        {contextOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};
