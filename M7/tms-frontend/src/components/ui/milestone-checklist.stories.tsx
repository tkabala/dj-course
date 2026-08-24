import React from 'react';
import { Meta, StoryObj } from '@storybook/react-vite';
import { MilestoneChecklist } from './milestone-checklist';

const meta: Meta<typeof MilestoneChecklist> = {
  title: 'UI/MilestoneChecklist',
  component: MilestoneChecklist,
  tags: ['autodocs'],
};

export default meta;

type Story = StoryObj<typeof MilestoneChecklist>;

export const Default: Story = {
  args: {
    milestones: [
      { id: '1', label: '50+ completed deliveries', achieved: true },
      { id: '2', label: 'On-time rate above 95%', achieved: true },
      { id: '3', label: 'Zero incidents this quarter', achieved: false },
      { id: '4', label: 'License valid for 6+ months', achieved: false },
    ],
  },
};

export const AllAchieved: Story = {
  args: {
    milestones: [
      { id: '1', label: '50+ completed deliveries', achieved: true },
      { id: '2', label: 'On-time rate above 95%', achieved: true },
      { id: '3', label: 'Zero incidents this quarter', achieved: true },
    ],
  },
};

export const NoneAchieved: Story = {
  args: {
    milestones: [
      { id: '1', label: '50+ completed deliveries', achieved: false },
      { id: '2', label: 'On-time rate above 95%', achieved: false },
    ],
  },
};
