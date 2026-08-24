import React from 'react';
import { Check } from 'lucide-react';
import { cn } from '@/lib/tailwind/utils';

export interface Milestone {
  id: string;
  label: string;
  achieved: boolean;
}

export interface MilestoneChecklistProps {
  milestones: Milestone[];
  className?: string;
}

export const MilestoneChecklist: React.FC<MilestoneChecklistProps> = ({ milestones, className }) => {
  return (
    <ul className={cn('space-y-2', className)}>
      {milestones.map((milestone) => (
        <li key={milestone.id} className="flex items-center gap-2">
          <span
            className={cn(
              'flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2 transition-colors',
              milestone.achieved ? 'bg-green-500 border-green-500' : 'border-gray-300 bg-white'
            )}
          >
            {milestone.achieved && <Check className="h-3 w-3 text-white" strokeWidth={3} />}
          </span>
          <span className={cn('text-sm', milestone.achieved ? 'text-gray-900' : 'text-gray-400')}>
            {milestone.label}
          </span>
        </li>
      ))}
    </ul>
  );
};
