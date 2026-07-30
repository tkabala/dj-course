import React from 'react';
import { useAtom } from 'jotai';
import { Search, Filter } from 'lucide-react';
import {
  routePlannerSearchTermAtom,
  routePlannerStatusFilterAtom,
} from '../route-planner.store';

export const RouteFilters: React.FC = () => {
  const [searchTerm, setSearchTerm] = useAtom(routePlannerSearchTermAtom);
  const [statusFilter, setStatusFilter] = useAtom(routePlannerStatusFilterAtom);

  return (
    <>
      <div className="relative">
        <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          placeholder="Search routes..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        />
      </div>

      <div className="relative">
        <Filter className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm appearance-none"
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="completed">Completed</option>
          <option value="planned">Planned</option>
          <option value="delayed">Delayed</option>
        </select>
      </div>
    </>
  );
};
